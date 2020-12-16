#![warn(missing_docs)]
#![doc(test(no_crate_inject))]
#![doc(test(attr(deny(unused, future_incompatible))))]

//! This crate provides an implementation of Reconstructability Analysis, as described by these
//! papers:
//!
//! - Zwick, [An Overview of Reconstructability Analysis][overview], 2004
//! - Zwick, [Wholes and Parts in General Systems Methodology][wholes], 2001
//!
//! [overview]: https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1022&context=sysc_fac
//! [wholes]: https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1026&context=sysc_fac
//!
//! Where those papers didn't describe some details, I also read the existing C++ implementation,
//! [OCCAM][], and the [OCCAM manual][].
//!
//! [OCCAM]: https://github.com/occam-ra/occam
//! [OCCAM manual]: https://occam.readthedocs.io/en/latest/

pub use sorted_iter;

use lasso::{LargeSpur, MicroSpur, MiniSpur, Spur};
use smallvec::SmallVec;
use sorted_iter::assume::{AssumeSortedByItemExt, AssumeSortedByKeyExt};
use sorted_iter::sorted_iterator::SortedByItem;
use sorted_iter::{multiway_union, SortedIterator, SortedPairIterator};
use statrs::distribution::{ChiSquared, Univariate};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::f64::consts::LN_2;
use std::iter;
use std::mem::swap;

/// Types which can be used in a [`VariableSet`].
pub trait VariableId: Sized + Copy + std::hash::Hash + Ord {
    /// SmallVec contains two `usize` fields which overlap with the inline vector, so variable sets
    /// will have minimum size if this array occupies the same number of bytes.
    ///
    /// It can be declared like this for any implementation, or you can have the [`variable_id!`]
    /// macro do it for you.
    ///
    /// ```ignore
    /// use std::mem::size_of;
    /// type SmallArray = [Self; 2 * size_of::<usize>() / size_of::<Self>()];
    /// ```
    type SmallArray: smallvec::Array<Item = Self> + Clone + std::fmt::Debug + std::hash::Hash + Ord;
}

/// Generates implementations of the [`VariableId`] trait which set the associated `SmallArray`
/// type to the biggest array that will fit within a [`SmallVec`][smallvec::SmallVec]'s minimum
/// size.
///
/// It also generates a test with the given `$testname` that checks that the generated definition
/// is as small as the smallest `SmallVec`.
///
/// For example, this library provides implementations for the basic unsigned integer types using
/// this declaration:
///
/// ```ignore
/// variable_id![unsigned_id_size, u8, u16, u32, u64, usize];
/// ```
#[macro_export]
macro_rules! variable_id {
    ($testname:ident, $($t:ty),*) => {
        $(
            impl $crate::VariableId for $t {
                type SmallArray = [
                    Self;
                    2 * ::std::mem::size_of::<usize>() / ::std::mem::size_of::<Self>()
                ];
            }
        )*

        #[cfg(test)]
        #[test]
        fn $testname() {
            use $crate::VariableSet;
            use smallvec::SmallVec;
            use std::mem::size_of;
            $(
                assert_eq!(
                    size_of::<VariableSet<$t>>(),
                    size_of::<SmallVec<[(); 0]>>()
                );
            )*
        }
    };
}

variable_id![lasso_id_size, LargeSpur, Spur, MiniSpur, MicroSpur];
variable_id![unsigned_id_size, u8, u16, u32, u64, usize];
variable_id![signed_id_size, i8, i16, i32, i64, isize];

/// A set of variables.
///
/// This implementation avoids heap allocations for sets containing a number of variables smaller
/// than the length of [`VariableId::SmallArray`].
#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VariableSet<V: VariableId>(SmallVec<V::SmallArray>);

impl<V: VariableId> VariableSet<V> {
    /// Creates a variable set containing the specified variables.
    ///
    /// It's okay if the provided slice contains duplicates.
    pub fn new(ids: &[V]) -> Self {
        let mut v = SmallVec::from_slice(ids);
        v.sort_unstable();
        v.dedup();
        VariableSet(v)
    }

    /// The number of variables in the set.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns an iterator over the variables which appear in this set.
    ///
    /// ```
    /// use reconstructability::VariableSet;
    ///
    /// let abc = VariableSet::new(&[2, 3, 1]);
    /// let mut it = abc.iter();
    /// assert_eq!(it.next(), Some(1));
    /// assert_eq!(it.next(), Some(2));
    /// assert_eq!(it.next(), Some(3));
    /// assert_eq!(it.next(), None);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = V> + SortedByItem + '_ {
        self.0.iter().copied().assume_sorted_by_item()
    }

    /// Returns `true` if `other` contains every variable that `self` does.
    ///
    /// ```
    /// use reconstructability::VariableSet;
    /// let nil = VariableSet::new(&[]);
    /// let one = VariableSet::new(&[1]);
    ///
    /// assert!(nil.is_subset(&one));
    /// assert!(nil.is_subset(&nil));
    /// assert!(one.is_subset(&one));
    /// assert!(!one.is_subset(&nil));
    /// ```
    pub fn is_subset(&self, other: &Self) -> bool {
        self.len() <= other.len() && self.iter().intersection(other.iter()).eq(self.iter())
    }

    /// Returns `true` if `self` contains every variable that `other` does.
    ///
    /// ```
    /// use reconstructability::VariableSet;
    /// let nil = VariableSet::new(&[]);
    /// let one = VariableSet::new(&[1]);
    ///
    /// assert!(!nil.is_superset(&one));
    /// assert!(nil.is_superset(&nil));
    /// assert!(one.is_superset(&one));
    /// assert!(one.is_superset(&nil));
    /// ```
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Returns an iterator over every subset of this variable set that has one less variable in
    /// it.
    fn remove_one_variable(
        &self,
    ) -> impl DoubleEndedIterator<Item = Self> + ExactSizeIterator + iter::FusedIterator + '_ {
        // Working from the end backward produces the subsets in lexicographic order, which reduces
        // copying if all of these subsets are added to a model.
        (0..self.len()).rev().map(move |remove_idx| {
            let mut selected = VariableSet(SmallVec::with_capacity(self.len() - 1));
            selected.0.extend_from_slice(&self.0[..remove_idx]);
            selected.0.extend_from_slice(&self.0[remove_idx + 1..]);
            selected
        })
    }
}

impl<V: VariableId + std::fmt::Debug> std::fmt::Debug for VariableSet<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.0.iter()).finish()
    }
}

impl<V: VariableId> iter::FromIterator<V> for VariableSet<V> {
    /// Creates a variable set containing the specified variables.
    ///
    /// It's okay if the provided iterator contains duplicates.
    fn from_iter<I: IntoIterator<Item = V>>(iter: I) -> Self {
        let mut v = SmallVec::from_iter(iter);
        v.sort_unstable();
        v.dedup();
        VariableSet(v)
    }
}

/// A sparse contingency table containing either frequencies or probabilities, which may come from
/// either observed events or maximum-likelihood estimates.
///
/// Variables are assumed to be binary. That is, an event is in a variable's category if that
/// variable appears in the variable-set key for that event, and not otherwise.
///
/// As the number of variables increase, odds are that the total number of observations is
/// significantly less than the size of the possible state space. So this type uses a sparse
/// representation, in which not all possible combinations of the variables need to be present in
/// the table. Missing elements are treated as zero.
///
/// # Panics
///
/// Various methods may panic if any cell in the table is negative or zero.
#[derive(Clone, PartialEq)]
pub struct Table<V: VariableId> {
    /// Each cell of the table is represented with a key identifying the set of variables which
    /// were true for that group of observations, and a value which is the number of events which
    /// were observed with that particular configuration of variables.
    //
    // Note: I considered using a HashMap to build a table (because we often need random access
    // during construction), but then converting it into a Vec before using it because we almost
    // always just need to iterate over all the cells in the table. However:
    // - iterating over a HashMap is almost as fast as over a vector;
    // - a HashMap is almost as small as a vector of key/value pairs, especially after calling
    //   `shrink_to_fit`;
    // - sometimes we still need random access;
    // - keeping all tables in a single representation is just simpler.
    pub raw_data: HashMap<VariableSet<V>, f64>,
}

/// Summary statistics for a single [`Table`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct TableSummary {
    /// The [Shannon entropy][] of the table, measured in bits. This is normalized to produce the
    /// same result whether the table contains probabilities or frequencies.
    ///
    /// [Shannon entropy]: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    pub uncertainty: f64,

    /// The total count across all cells in the table. If this table contains probabilities, the
    /// `sample_size` should be approximately 1.0.
    pub sample_size: f64,
}

impl iter::FromIterator<f64> for TableSummary {
    /// Creates a summary for a table whose non-zero cells are provided by the given iterator.
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        let mut summary = TableSummary {
            uncertainty: 0.0,
            sample_size: 0.0,
        };
        for count in iter {
            summary.uncertainty -= count * count.log2();
            summary.sample_size += count;
        }
        if summary.sample_size > 0.0 {
            summary.uncertainty /= summary.sample_size;
            summary.uncertainty += summary.sample_size.log2();
        }
        summary
    }
}

impl<V> std::fmt::Debug for Table<V>
where
    V: VariableId,
    VariableSet<V>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.raw_data.iter()).finish()
    }
}

impl<V: VariableId> Table<V> {
    /// Creates an empty table.
    pub fn new() -> Self {
        Table {
            raw_data: HashMap::new(),
        }
    }

    /// Creates an empty table with the specified capacity.
    ///
    /// The table will be able to hold at least `capacity` elements without reallocating. If
    /// `capacity` is 0, the table will not allocate.
    pub fn with_capacity(capacity: usize) -> Self {
        Table {
            raw_data: HashMap::with_capacity(capacity),
        }
    }

    /// Add the given frequency or probability to the specified cell.
    ///
    /// If the cell had not previously been set, it's treated like it was 0.
    ///
    /// # Panics
    ///
    /// This method panics if the count is not finite and positive.
    pub fn add_cell(&mut self, state: VariableSet<V>, count: f64) -> &mut Self {
        assert!(count > 0.0 && count.is_finite());
        *self.raw_data.entry(state).or_insert(0.0) += count;
        self
    }

    /// Returns a new table that is the projection of this table on a smaller set of variables.
    ///
    /// ```
    /// use reconstructability::{Table, VariableSet};
    ///
    /// let ab = VariableSet::new(&[1, 2]);
    /// let a = VariableSet::new(&[1]);
    /// let b = VariableSet::new(&[2]);
    /// let nil = VariableSet::new(&[]);
    ///
    /// let mut table = Table::new();
    /// table.add_cell(nil.clone(), 0.125);
    /// table.add_cell(a.clone(), 0.25);
    /// table.add_cell(b.clone(), 0.375);
    /// table.add_cell(ab.clone(), 0.25);
    ///
    /// let project_a = table.project(&a);
    /// assert_eq!(project_a, *Table::new().add_cell(nil.clone(), 0.5).add_cell(a.clone(), 0.5));
    ///
    /// let project_b = table.project(&b);
    /// assert_eq!(project_b, *Table::new().add_cell(nil.clone(), 0.375).add_cell(b.clone(), 0.625));
    /// ```
    pub fn project(&self, on: &VariableSet<V>) -> Table<V> {
        // In theory, there are 2**on.len() states in this projected table, so we can preallocate
        // storage for them. But don't preallocate too much capacity; the input data is typically
        // sparse so it may be that many of the possible projected states are empty. Worse, the number
        // of possible states may far exceed available memory, so we can't afford to be over-eager.
        let mut projection = Table::with_capacity((1 << on.len().min(8)).max(self.raw_data.len()));

        // Reuse the same heap allocation for every projection to avoid hammering the allocator.
        let mut projected_state = Vec::with_capacity(on.len());
        for (state, count) in self.raw_data.iter() {
            projected_state.extend(state.iter().intersection(on.iter()));
            projection.add_cell(VariableSet::new(&projected_state), *count);
            projected_state.clear();
        }
        projection
    }

    /// ```
    /// use reconstructability::{Table, VariableSet};
    ///
    /// let ab = VariableSet::new(&[1, 2]);
    /// let a = VariableSet::new(&[1]);
    /// let b = VariableSet::new(&[2]);
    /// let nil = VariableSet::new(&[]);
    ///
    /// let table = Table::new()
    ///     .add_cell(nil.clone(), 1.0)
    ///     .add_cell(a.clone(), 7.0)
    ///     .compose_with(Table::new()
    ///         .add_cell(nil.clone(), 2.0)
    ///         .add_cell(b.clone(), 6.0));
    ///
    /// assert_eq!(
    ///     &table,
    ///     Table::new()
    ///     .add_cell(nil.clone(), 0.25)
    ///     .add_cell(a.clone(), 1.75)
    ///     .add_cell(b.clone(), 0.75)
    ///     .add_cell(ab.clone(), 5.25)
    /// );
    /// ```
    pub fn compose_with(&self, other: &Self) -> Table<V> {
        let mut composition = Table::with_capacity(self.raw_data.len() * other.raw_data.len());
        let mut sample_size = 0.0;

        // Reuse the same heap allocation for every projection to avoid hammering the allocator.
        let mut composed_state = Vec::new();
        for (state_a, count_a) in self.raw_data.iter() {
            sample_size += count_a;
            for (state_b, count_b) in other.raw_data.iter() {
                composed_state.extend(state_a.iter().union(state_b.iter()));
                composition.add_cell(VariableSet::new(&composed_state), count_a * count_b);
                composed_state.clear();
            }
        }

        assert_eq!(sample_size, other.raw_data.values().copied().sum());
        for cell in composition.raw_data.values_mut() {
            *cell /= sample_size;
        }
        composition
    }

    /// Compute a [`TableSummary`] over the current contents of this table.
    ///
    /// ```
    /// use reconstructability::{Table, VariableSet};
    ///
    /// let mut builder = Table::new();
    /// assert_eq!(builder.summary().uncertainty, 0.0);
    ///
    /// builder.add_cell(VariableSet::new(&[]), 1.0);
    /// assert_eq!(builder.summary().uncertainty, 0.0);
    ///
    /// builder.add_cell(VariableSet::new(&[1]), 1.0);
    /// assert_eq!(builder.summary().uncertainty, 1.0);
    ///
    /// builder.add_cell(VariableSet::new(&[2]), 1.0);
    /// builder.add_cell(VariableSet::new(&[1, 2]), 1.0);
    /// assert_eq!(builder.summary().uncertainty, 2.0);
    /// ```
    pub fn summary(&self) -> TableSummary {
        self.raw_data.values().copied().collect()
    }

    /// Shrinks the capacity of the table as much as possible.
    ///
    /// If you're going to access the contents of this table a lot, it's a good idea to call this
    /// method after you finish constructing it but before you start using it.
    pub fn shrink_to_fit(&mut self) {
        self.raw_data.shrink_to_fit();
    }
}

/// A model is a set of [`VariableSet`]s, called "relations", where no relation is a subset of any
/// other in the same model.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Model<V: VariableId> {
    relations: Vec<VariableSet<V>>,
}

impl<V: VariableId> Model<V> {
    /// Creates a new empty model.
    pub fn new() -> Self {
        Model {
            relations: Vec::new(),
        }
    }

    /// For a given set of variables, creates the model which asserts that every variable is
    /// independent of all the others.
    pub fn independence(variables: &VariableSet<V>) -> Self {
        Model {
            relations: variables
                .0
                .iter()
                .map(|variable| VariableSet::new(std::slice::from_ref(variable)))
                .collect(),
        }
    }

    /// Returns an iterator over the relations which appear in this model.
    ///
    /// ```
    /// use reconstructability::{Model, VariableSet};
    ///
    /// let mut model = Model::new();
    /// model.add_relation(VariableSet::new(&[2]));
    /// model.add_relation(VariableSet::new(&[3, 1]));
    ///
    /// let mut it = model.iter();
    /// assert_eq!(it.next(), Some(&VariableSet::new(&[1, 3])));
    /// assert_eq!(it.next(), Some(&VariableSet::new(&[2])));
    /// assert_eq!(it.next(), None);
    /// ```
    pub fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = &VariableSet<V>> + ExactSizeIterator + iter::FusedIterator + '_
    {
        self.relations.iter()
    }

    // This implementation relies on the relations being sorted in descending order by number of
    // variables. Break ties by VariableSet's natural order so that there's a canonical order and
    // derived Eq/Ord/Hash just work.
    fn sort_by(a: &VariableSet<V>, b: &VariableSet<V>) -> Ordering {
        b.len().cmp(&a.len()).then_with(|| a.cmp(b))
    }

    /// Adds a relation to this model if it is not a subset of an existing relation, and removes
    /// relations which are a subset of this one.
    ///
    /// ```
    /// use reconstructability::{Model, VariableSet};
    ///
    /// let ab = VariableSet::new(&[1, 2]);
    /// let a = VariableSet::new(&[1]);
    /// let b = VariableSet::new(&[2]);
    /// let nil = VariableSet::new(&[]);
    ///
    /// assert_eq!(
    ///     Model::new().add_relation(nil.clone()),
    ///     &mut Model::new()
    /// );
    ///
    /// assert_eq!(
    ///     Model::new().add_relation(a.clone()).add_relation(nil.clone()),
    ///     Model::new().add_relation(a.clone())
    /// );
    ///
    /// assert_eq!(
    ///     Model::new().add_relation(a.clone()).add_relation(b.clone()),
    ///     Model::new().add_relation(b.clone()).add_relation(a.clone())
    /// );
    ///
    /// assert_eq!(
    ///     Model::new().add_relation(ab.clone()).add_relation(a.clone()),
    ///     Model::new().add_relation(ab.clone())
    /// );
    ///
    /// assert_eq!(
    ///     Model::new().add_relation(a.clone()).add_relation(ab.clone()),
    ///     Model::new().add_relation(ab.clone())
    /// );
    /// ```
    pub fn add_relation(&mut self, relation: VariableSet<V>) -> &mut Self {
        let len = relation.len();

        // If binary search finds the specified relation, then we don't need to add it again.
        if let Err(insert_at) = self
            .relations
            .binary_search_by(|probe| Model::sort_by(probe, &relation))
        {
            // If there's a larger relation which contains this one, don't add this one.
            let is_subset = self.relations[..insert_at]
                .iter()
                .take_while(|r| len < r.len())
                .any(|r| relation.is_subset(r));
            if !is_subset && len > 0 {
                // If there are any smaller relations subsumed by this one, keep everything except
                // those relations.
                self.relations
                    .retain(|r| len <= r.len() || !r.is_subset(&relation));

                // We retained everything of this relation's size or bigger, so at least up to the
                // point of insertion nothing has changed and the position we found with binary
                // search is still valid.
                self.relations.insert(insert_at, relation);
            }
        }
        self
    }

    /// An iterator producing every model which is one step down the lattice of structures.
    ///
    /// See the papers cited in the crate-level documentation for an explanation of the lattice of
    /// structures.
    ///
    /// Because these models only support binary variables, every model returned has exactly one
    /// degree of freedom less than the current model.
    ///
    /// ```
    /// use reconstructability::{Model, VariableSet};
    ///
    /// let mut abc = Model::new();
    /// abc.add_relation(VariableSet::new(&[1, 2, 3]));
    ///
    /// let ab = VariableSet::new(&[1, 2]);
    /// let ac = VariableSet::new(&[1, 3]);
    /// let bc = VariableSet::new(&[2, 3]);
    ///
    /// let mut df1 = abc.less_complex();
    /// let mut ab_ac_bc = Model::new();
    /// ab_ac_bc.add_relation(ab.clone()).add_relation(ac.clone()).add_relation(bc.clone());
    ///
    /// assert_eq!(df1.next(), Some(ab_ac_bc));
    /// assert_eq!(df1.next(), None);
    /// ```
    pub fn less_complex(
        &self,
    ) -> impl DoubleEndedIterator<Item = Self> + ExactSizeIterator + iter::FusedIterator + '_ {
        // A relation of only one variable can't get any simpler, but any other relation in this
        // model is fair game. If all the relations are single-variable, this is the independence
        // model and has no simpler models.
        let models = self
            .relations
            .iter()
            .rposition(|relation| relation.0.len() > 1)
            .map_or(0, |index| index + 1);

        (0..models).into_iter().map(move |simplify_at| {
            // Copy all the relations except our current simplification target into a new model.
            let mut model = Model::new();
            let (before, after) = self.relations.split_at(simplify_at);
            let (removed, after) = after.split_first().unwrap();
            model.relations.extend_from_slice(before);
            model.relations.extend_from_slice(after);

            for reduced in removed.remove_one_variable() {
                model.add_relation(reduced);
            }
            model
        })
    }

    /// An iterator producing every model which is one step up the lattice of structures.
    ///
    /// See the papers cited in the crate-level documentation for an explanation of the lattice of
    /// structures.
    ///
    /// Because these models only support binary variables, every model returned has exactly one
    /// degree of freedom more than the current model.
    ///
    /// ```
    /// use reconstructability::{Model, VariableSet};
    ///
    /// let mut abc = Model::new();
    /// abc.add_relation(VariableSet::new(&[1, 2, 3]));
    ///
    /// let ab = VariableSet::new(&[1, 2]);
    /// let ac = VariableSet::new(&[1, 3]);
    /// let bc = VariableSet::new(&[2, 3]);
    ///
    /// let mut ab_ac_bc = Model::new();
    /// ab_ac_bc.add_relation(ab.clone()).add_relation(ac.clone()).add_relation(bc.clone());
    ///
    /// let mut df1 = ab_ac_bc.more_complex();
    ///
    /// assert_eq!(df1.next(), Some(abc));
    /// assert_eq!(df1.next(), None);
    /// ```
    pub fn more_complex(&self) -> impl Iterator<Item = Self> + iter::FusedIterator + '_ {
        // `less_complex` picks a relation from the model and replaces it with all relations that
        // have one variable less. So to find a more complex model, we need to find a relation such
        // that:
        //
        // 1. all its one-smaller subsets are already subsets of this model, and
        // 2. it isn't a subset of any relation in the model.
        //
        // Adding that as a new relation creates a more complex model, one level up the lattice of
        // structures.
        //
        // A useful alternative formulation of the goal is to look for a combination of N variables
        // which occur across N relations, where each relation is missing a different variable, or
        // equivalently each variable is missing in a different relation. This is necessary and
        // sufficient for condition #1 above. It is also necessary and sufficient for a limited
        // form of condition #2: the result is not a subset of any of the selected relations, but
        // may be a subset of some other relation in the model.
        //
        // Since a variable can only occur once in a relation, it can't occur more than N times in
        // N relations. And if a variable occurs less than N-1 times in N relations, then it not
        // only isn't part of a candidate for this size, but adding more relations can't make it
        // part of a candidate either.
        //
        // So we keep track of those variables which have occurred either in every one of the
        // currently-selected N relations, or in all but one of the relations. For the latter, we
        // also keep track of which relation each variable is missing from.

        struct State<I, V: VariableId> {
            relations: I,
            missing_none: VariableSet<V>,
            missing_one: Vec<(V, usize)>,
        }
        let mut stack = Vec::new();
        stack.push(State {
            relations: self.relations.iter(),
            missing_none: VariableSet::new(&[]),
            missing_one: Vec::new(),
        });

        let mut candidates = HashSet::new();

        while let Some((cur, prev)) = stack.split_last_mut() {
            if let Some(relation) = cur.relations.next() {
                // For a variable to be in one of the two sets after adding the Nth relation,
                // it must have appeared in at least N-2 of the previous N-1 relations. But
                // that means that the first and second relations are special because they
                // require -1 or 0 prior occurrences, respectively, which is true for all
                // variables.

                let relation_idx = prev.len();
                if relation_idx == 0 {
                    // We're adding the first relation to the selection, so by definition, the
                    // variables in this relation have appeared in every relation so far and we
                    // haven't seen any variables that are missing from any relation so far.
                    let relations = cur.relations.clone();
                    stack.push(State {
                        relations,
                        missing_none: relation.clone(),
                        missing_one: Vec::new(),
                    });
                } else if relation_idx == 1 {
                    // We're adding the second relation to the selection. The variables from the
                    // first relation are in missing_none. Any variable that's also in this
                    // relation stays in missing_none; otherwise annotate it by whether it was
                    // missing in relation 0 or relation 1.
                    let mut missing_none = VariableSet::new(&[]);
                    let mut missing_one = Vec::new();
                    let old = cur.missing_none.iter().pairs();
                    let new = relation.iter().pairs();
                    for (v, (old, new)) in old.outer_join(new) {
                        match (old.is_some(), new.is_some()) {
                            (true, true) => missing_none.0.push(v),
                            (true, false) => missing_one.push((v, 1)),
                            (false, true) => missing_one.push((v, 0)),
                            (false, false) => unreachable!(),
                        }
                    }
                    debug_assert!(missing_one.iter().any(|(_, idx)| *idx == 0));
                    debug_assert!(missing_one.iter().any(|(_, idx)| *idx == 1));

                    let relations = cur.relations.clone();
                    stack.push(State {
                        relations,
                        missing_none,
                        missing_one,
                    });
                } else {
                    // Every subsequent relation we add lands here.
                    let mut missing_none = VariableSet::new(&[]);
                    let mut missing_one = Vec::new();

                    // As a minor optimization, don't heap-allocate this vector unless we have more
                    // relations than the number of bits SmallVec can store inline for free.
                    let mut supported_relations = SmallVec::<[usize; 2]>::new();
                    const BITS: usize = std::mem::size_of::<usize>() * 8;
                    let relation_count = relation_idx + 1;
                    supported_relations.resize((relation_count + BITS - 1) / BITS, 0);
                    if relation_count % BITS != 0 {
                        // Set the bits that don't correspond to any relation so later we can
                        // simply check whether all bits are set.
                        *supported_relations.last_mut().unwrap() = !0 << (relation_count % BITS);
                    }

                    let none = cur.missing_none.iter().pairs();
                    let one = cur.missing_one.iter().copied().assume_sorted_by_key();
                    let new = relation.iter().pairs();
                    for (v, ((none, one), new)) in none.outer_join(one).left_join(new) {
                        // outer_join ensures that at least one argument is Some, and we have
                        // an invariant that a variable is never in both sets, so we should
                        // have exactly one Some argument.
                        debug_assert_ne!(none.is_some(), one.is_some());

                        let missing_from = match (new, one) {
                            // Any variable present in the current relation stays in whichever
                            // set it was previously in.
                            (Some(_), None) => {
                                missing_none.0.push(v);
                                continue;
                            }
                            (Some(_), Some(idx)) => idx,

                            // Any variable missing from the relation either gets moved from
                            // missing_none to missing_one, or gets removed from missing_one.
                            (None, None) => relation_idx,
                            (None, Some(_)) => continue,
                        };
                        missing_one.push((v, missing_from));
                        supported_relations[missing_from / BITS] |= 1 << (missing_from % BITS);
                    }

                    // Every relation in the selected set must be missing at least one of the
                    // selected variables. Otherwise, the selected variables are a subset of some
                    // relation, so adding a new relation with those variables won't yield a more
                    // complex model. Adding more relations to the selection can only shrink the
                    // set of selected variables, so any extension of this state would also be a
                    // subset of some relation.
                    //
                    // Because a relation can't be equal to another relation in the same model,
                    // this condition is always satisfied for N <= 2, which is why the previous two
                    // cases unconditionally push a new state.
                    if supported_relations.into_iter().all(|w| w == !0) {
                        let relations = cur.relations.clone();
                        stack.push(State {
                            relations,
                            missing_none,
                            missing_one,
                        });
                    }
                }
            } else {
                // At this point we've checked all the extensions of this state that we're
                // going to and we just have to check this state itself. Most fields of State
                // are only useful when extending the subset of relations, so drop them now.
                let State {
                    mut missing_one, ..
                } = stack.pop().unwrap();

                if missing_one.is_empty() {
                    // This can only happen when we have fewer than two relations selected. The
                    // code below would give the correct result (no candidates generated) but we
                    // can short-circuit some allocations.
                    continue;
                }

                // Regroup the selected variables according to which relation they appeared in.
                missing_one.sort_unstable_by_key(|(v, idx)| (*idx, *v));

                // Every selected relation must be missing at least one variable, but some
                // relations might be missing multiple variables. Generate every subset of the
                // variables such that each relation lacks exactly one.
                let mut candidate = VariableSet(SmallVec::with_capacity(stack.len()));
                let mut iters = Vec::with_capacity(stack.len());
                iters.push(missing_one.into_iter());
                while let Some(iter) = iters.last_mut() {
                    if let Some((v, idx)) = iter.next() {
                        match idx.cmp(&candidate.len()) {
                            Ordering::Equal => {
                                candidate.0.push(v);
                                let iter = iter.clone();
                                iters.push(iter);
                                continue;
                            }
                            Ordering::Less => {
                                // The previous variable came from a relation that had multiple
                                // possibilities, so we're skipping past the other choices now.
                                debug_assert_eq!(idx + 1, candidate.len());
                                continue;
                            }
                            Ordering::Greater => {
                                // Stop trying variations for this relation once we've exhausted
                                // all its choices and fallen into the next relation. This is like
                                // reaching the overall end of the missing_one list, except at this
                                // point we haven't added variables from all the relations yet, so
                                // this isn't a complete candidate.
                            }
                        }
                    } else if candidate.len() >= stack.len() {
                        debug_assert_eq!(candidate.len(), stack.len());
                        let mut tmp = candidate.clone();
                        tmp.0.sort_unstable();
                        candidates.insert(tmp);
                    }
                    candidate.0.pop();
                    iters.pop();
                }
            }
        }

        candidates.into_iter().filter_map(move |relation| {
            let mut model = self.clone();
            model.add_relation(relation);
            if &model != self {
                Some(model)
            } else {
                None
            }
        })
    }

    /// Computes a plan for how to evaluate this model against the observed data.
    ///
    /// Every variable must contribute to the final result exactly once. When two relations in the
    /// model include the same variable, the effect of such shared variables must be removed from
    /// the result.
    ///
    /// Any set of relations which contains a loop must be jointly evaluated using an expensive
    /// Iterative Proportional Fit algorithm. IPF is asymptotically expensive:
    ///
    /// - IPF needs `O(2^n)` space, where `n` is the number of variables.
    /// - IPF needs `O(r*2^n)` time, where `r` is the number of relations in the model, which
    ///   itself is `O(2^(2^n))` (I think?).
    ///
    /// By contrast, loopless models can be evaluated algebraically using only [`TableSummary`]s,
    /// one for each relation and one for each of their shared subsets.
    ///
    /// So it's worth spending some time analyzing the model to find every relation which can be
    /// evaluated algebraically, and only use IPF for the minimum possible group of relations that
    /// require it.
    pub fn plan(&self) -> QueryPlan<V> {
        let mut reduced = self
            .relations
            .iter()
            .map(|relation| (relation.clone(), vec![(relation.clone(), 0)]))
            .collect::<HashMap<_, _>>();

        let mut loopless = Vec::new();
        let mut new_groups = HashMap::new();
        let mut occurrences = HashMap::new();

        loop {
            for relation in reduced.keys() {
                for variable in relation.0.iter().copied() {
                    *occurrences.entry(variable).or_insert(0) += 1;
                }
            }

            occurrences.retain(|_, count| *count == 1);
            if occurrences.is_empty() {
                break;
            }

            for (mut relation, mut plan) in reduced.drain() {
                relation.0.retain(|v| !occurrences.contains_key(v));
                if relation.0.is_empty() {
                    loopless.append(&mut plan);
                } else {
                    new_groups
                        .entry(relation)
                        .or_insert_with(Vec::new)
                        .push(plan);
                }
            }

            for (relation, mut plans) in new_groups.drain() {
                debug_assert!(plans.len() >= 2);
                let mut plan = plans.pop().unwrap();

                let duplicates = plans.len();
                for mut more_plan in plans {
                    plan.append(&mut more_plan);
                }

                let mut subsumed = false;
                reduced.retain(|k, v| {
                    if !subsumed {
                        match k.len().cmp(&relation.len()) {
                            Ordering::Less => {
                                if k.is_subset(&relation) {
                                    plan.append(v);
                                    plan.push((k.clone(), 1));
                                    return false;
                                }
                            }
                            Ordering::Greater => {
                                if relation.is_subset(k) {
                                    v.append(&mut plan);
                                    v.push((relation.clone(), 1 + duplicates));
                                    subsumed = true;
                                }
                            }
                            Ordering::Equal => {
                                // This can't be either a strict subset or a strict superset of the
                                // current relation, and it can't be the same set because the keys
                                // of new_groups are unique. Therefore it's unrelated; leave it
                                // alone.
                            }
                        }
                    }
                    true
                });

                if !subsumed {
                    plan.push((relation.clone(), duplicates));
                    let old = reduced.insert(relation, plan);
                    debug_assert!(old.is_none());
                }
            }

            occurrences.clear();
        }

        loopless.shrink_to_fit();

        // Now group loop relations such that each group shares no variables with any other group.
        // TODO: Identify relations which connect two loops but are not part of a loop.
        let mut disjoint_loop_vars = DisjointSets::new();
        for common in reduced.keys() {
            let first = common.0[0];
            for v in common.0[1..].iter().copied() {
                disjoint_loop_vars.union(first, v)
            }
        }

        let mut loops = HashMap::new();
        for (common, plan) in reduced {
            loops
                .entry(disjoint_loop_vars.find(common.0[0]))
                .or_insert_with(Vec::new)
                .extend(plan.into_iter().filter_map(|(relation, count)| {
                    if count == 0 {
                        Some(relation)
                    } else {
                        None
                    }
                }));
        }

        QueryPlan {
            loops: loops.into_iter().map(|(_, v)| v).collect(),
            loopless,
        }
    }
}

/// An evaluation plan produced by [`Model::plan`].
#[derive(Clone, Debug)]
pub struct QueryPlan<V: VariableId> {
    loops: Vec<Vec<VariableSet<V>>>,
    loopless: Vec<(VariableSet<V>, usize)>,
}

impl<V: VariableId> QueryPlan<V> {
    /// Computes the total degrees of freedom for the model that produced this plan.
    ///
    /// Asymptotically, this takes `O(r^2)` time, where `r` is the number of relations in the
    /// model. But it's faster for the loopless subset of relations than for the loops.
    pub fn degrees_of_freedom(&self) -> f64 {
        let mut total = 0.0;

        for group in self.loops.iter() {
            let mut relations = group.iter();
            while let Some(a) = relations.next() {
                total += (a.0.len() as f64).exp2() - 1.0;
                for b in relations.clone() {
                    total -= (a.iter().intersection(b.iter()).count() as f64).exp2() - 1.0;
                }
            }
        }

        for (relation, duplicates) in self.loopless.iter().rev() {
            let local_df = (relation.0.len() as f64).exp2() - 1.0;
            if *duplicates == 0 {
                total += local_df;
            } else {
                total -= local_df * (*duplicates as f64);
            }
        }

        total
    }

    /// Computes the [Shannon entropy][] for the model that produced this plan, measured in bits,
    /// given a table of observed events.
    ///
    /// [Shannon entropy]: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    ///
    /// This uses Iterated Proportional Fit for each loop, and algebraic results for the remaining
    /// relations. See [`Model::plan`] for more details.
    pub fn uncertainty(&self, table: &Table<V>) -> f64 {
        let mut uncertainty = 0.0;

        let mut projections = Vec::new();
        for relation_loop in self.loops.iter() {
            projections.extend(
                relation_loop
                    .iter()
                    .map(|relation| (relation, table.project(relation))),
            );
            uncertainty += iterative_proportional_fit(&projections, 255, 0.2)
                .summary()
                .uncertainty;
            projections.clear();
        }

        for (relation, factor) in self.loopless_factors() {
            let projection = table.project(relation).summary().uncertainty;
            uncertainty += projection * factor;
        }

        uncertainty
    }

    /// Returns `true` if the model which produced this plan contained any loops between relations.
    pub fn has_loops(&self) -> bool {
        !self.loops.is_empty()
    }

    /// An iterator over the relations in this plan which do not participate in any loops. Each
    /// relation is returned with a weight indicating how many times each of its variables should
    /// be counted.
    ///
    /// That weight will either be 1, or it will be a negative number for relations which are
    /// shared subsets of other relations in the model. In the abscense of loops, summing over the
    /// product of this weight with either uncertainty or degrees of freedom will give the correct
    /// result for the model overall.
    pub fn loopless_factors(&self) -> impl Iterator<Item = (&VariableSet<V>, f64)> {
        self.loopless.iter().map(|(relation, duplicates)| {
            (
                relation,
                if *duplicates == 0 {
                    1.0
                } else {
                    -(*duplicates as f64)
                },
            )
        })
    }
}

fn iterative_proportional_fit<V: VariableId>(
    projections: &[(&VariableSet<V>, Table<V>)],
    max_iter: u8,
    max_error: f64,
) -> Table<V> {
    // I've chosen to build up the result in a vector rather than a HashMap or Table because we
    // need to iterate over it a bunch of times, so even the small overhead for iteration over a
    // HashMap adds up.
    //
    // Also, due to the order in which `init` constructs the initial result vector, each table in
    // `projections` will be accessed in its iteration order as long as we preserve order in
    // `result`. That should improve cache locality.

    fn init<'a, V: VariableId>(
        result: &mut Vec<(Vec<&'a VariableSet<V>>, f64)>,
        keys: &mut Vec<&'a VariableSet<V>>,
        margins: &'a [(&'a VariableSet<V>, Table<V>)],
        seen: &VariableSet<V>,
        state: VariableSet<V>,
    ) {
        if let Some(((relation, margin), rest)) = margins.split_first() {
            let common: VariableSet<V> = seen.iter().intersection(relation.iter()).collect();
            let common_state: VariableSet<V> = common.iter().intersection(state.iter()).collect();
            let seen: VariableSet<V> = seen.iter().union(relation.iter()).collect();

            for current in margin.raw_data.keys() {
                if common
                    .iter()
                    .intersection(current.iter())
                    .eq(common_state.iter())
                {
                    keys.push(current);
                    let new_state = state.iter().union(current.iter());
                    init(result, keys, rest, &seen, new_state.collect());
                    keys.pop();
                }
            }
        } else {
            result.push((keys.clone(), 1.0));
        }
    }

    let mut result = Vec::new();
    init(
        &mut result,
        &mut Vec::new(),
        projections,
        &VariableSet::new(&[]),
        VariableSet::new(&[]),
    );

    let mut current_margin = HashMap::new();
    let mut next_margin = HashMap::new();
    for (keys, count) in result.iter() {
        *current_margin.entry(keys[0]).or_insert(0.0) += *count;
    }

    for _iteration in 0..max_iter {
        let mut error = 0.0;

        // Every iteration of this loop fits the table to one projection of the original data.
        // During the same pass, it also tallies up the revised margin over the relation that the
        // next iteration will use.
        for (idx, (_, margin)) in projections.iter().enumerate() {
            let next_idx = (idx + 1) % projections.len();

            for (keys, count) in result.iter_mut() {
                let key = keys[idx];
                let projection = margin.raw_data[key];
                let current = current_margin[key];

                error = (projection - current).abs().max(error);

                *count *= projection / current;
                debug_assert!(*count > 0.0 && count.is_finite());

                *next_margin.entry(keys[next_idx]).or_insert(0.0) += *count;
            }

            current_margin.clear();
            swap(&mut current_margin, &mut next_margin);
        }

        if error < max_error {
            dbg!(_iteration, error, max_error);
            break;
        }
    }

    let result_len = result.len();
    let raw_data: HashMap<_, _> = result
        .into_iter()
        .map(|(keys, count)| {
            let state: VariableSet<_> =
                multiway_union(keys.into_iter().map(|key| key.iter())).collect();
            (state, count)
        })
        .collect();
    assert_eq!(
        result_len,
        raw_data.len(),
        "iterative_proportional_fit generated duplicate variable-sets"
    );
    Table { raw_data }
}

#[derive(Clone, Copy, Debug)]
struct RawEvaluation {
    uncertainty: f64,
    degrees_of_freedom: f64,
}

impl RawEvaluation {
    fn log_likelihood(&self, sample_size: f64) -> f64 {
        // We have a set of observations of events and we're trying to pick a probability
        // distribution that would be most likely to have generated those observations. Each event
        // is identified with an assortment of categorical variables, and a particular combination
        // of categories has some (unknown) probability of occurring. That makes this a multinomial
        // distribution; if there were exactly two categories it would be a binomial distribution.
        //
        // The likelihood of a particular observation, given some guess about what the
        // probabilities might be of the various possible events, is exactly the result of
        // evaluating the probability mass function of the multinomial distribution parameterized
        // by the estimated event probabilities. That PMF is proportional to this expression:
        //
        // p1**x1 * p2**x2 * ... * pk**xk
        //
        // Where p1...pk are the estimated probabilities, and x1...xk are the observed event
        // counts.
        //
        // (There's also a factor of n!/(x1!...xk!) but as long as we only compare the differences
        // between log-likelihoods, or equivalently the ratio of likelihoods, that factor cancels
        // out since it's constant for a given set of observations. So we ignore it here, but be
        // aware that only differences in values computed from this "likelihood" are meaningful.)
        //
        // The estimated probabilities with the highest likelihood are exactly the observed
        // probabilities, corresponding to the saturation or "top" model. But simpler models may
        // also be pretty likely; that's what we're checking in reconstructability analysis.
        //
        // Taking the logarithm of that expression we get:
        //
        // x1*log(p1) + x2*log(p2) + ... + xk*log(pk)
        //
        // And if we've estimated the probabilities as equal to the observed probabilities, then we
        // can rewrite all the observed counts by multiplying the probability by the sample size,
        // so e.g. x1 becomes p1*N. Aftor factoring out the common N, we have simply sample_size
        // times uncertainty, except that uncertainty is measured in bits, so we have to change its
        // base to base-e nats to match the scale that other standard statistics equations assume.
        //
        // However, what about when the estimated probabilities don't equal the observed
        // probabilities because we're testing a simpler model? The key here is we only evaluate
        // maximum-likelihood estimates, which are of a special form. If we're testing the
        // independence model for a two-variable problem, then the estimated probability of A=1,B=1
        // is the observed probability of A=1 (across all values of B) times the observed
        // probability of B=1 (across all values of A), and similarly for the other states.
        //
        // X(A=1,B=1)*log(
        //     (X(A=1,B=0) + X(A=1,B=1)) * (X(A=0,B=1) + X(A=1,B=1))
        // ) + X(A=0,B=1)*log(
        //     (X(A=0,B=0) + X(A=0,B=1)) * (X(A=0,B=1) + X(A=1,B=1))
        // ) + ...
        //
        // is equivalent to:
        //
        // X(A=1,B=1)*log((X(A=1,B=0) + X(A=1,B=1))) +
        // X(A=1,B=1)*log((X(A=0,B=1) + X(A=1,B=1))) +
        // X(A=0,B=1)*log((X(A=0,B=0) + X(A=0,B=1))) +
        // X(A=0,B=1)*log((X(A=0,B=1) + X(A=1,B=1))) + ...
        //
        // If we group the terms where the log factor has the same argument, such as the second and
        // fourth term above, we end up with terms like this:
        //
        // (X(A=0,B=1) + X(A=1,B=1)) * log((X(A=0,B=1) + X(A=1,B=1)))
        //
        // and, delightfully, that is exactly the p*log(p) form of the uncertainty of the
        // independence model. This generalizes to models of intermediate complexity as well.
        sample_size * self.uncertainty * LN_2
    }

    fn akaike_information_criterion(&self, sample_size: f64) -> f64 {
        2.0 * (self.log_likelihood(sample_size) + self.degrees_of_freedom)
    }

    fn bayesian_information_criterion(&self, sample_size: f64) -> f64 {
        2.0 * self.log_likelihood(sample_size) + sample_size.ln() * self.degrees_of_freedom
    }
}

/// Holds the minimum state necessary for efficiently evaluating the quality of many candidate
/// models for the same data.
pub struct ModelEvaluator<'a, V: VariableId> {
    data: &'a Table<V>,
    sample_size: f64,
    variables: VariableSet<V>,
    top: RawEvaluation,
    bottom: RawEvaluation,
}

impl<V: VariableId> ModelEvaluator<'_, V> {
    /// Creates an evaluator for the given observed data.
    ///
    /// This method precomputes statistics for the top and bottom models in the lattice of
    /// structures, so that any other model can be quickly compared to those two references.
    ///
    /// It takes time proportional to number of cells in the data plus the number of variables.
    pub fn new<'a>(data: &'a Table<V>) -> ModelEvaluator<'a, V> {
        let mut variables = HashMap::new();

        let summary: TableSummary = data
            .raw_data
            .iter()
            .map(|(relation, count)| {
                // While summarizing the original data, also count how many observations each
                // variable occurs in. This is like computing each single-variable projection for
                // the independence model, but stores only the positive counts and makes only one
                // pass over the data.
                for variable in relation.0.iter() {
                    *variables.entry(*variable).or_insert(0.0) += *count;
                }
                *count
            })
            .collect();

        let mut bottom_uncertainty = 0.0;
        let variables: VariableSet<V> = variables
            .into_iter()
            .map(|(variable, count)| {
                // Now that we have all the single-variable projections, convert to probabilities
                // and also find the complementary probability for each variable.
                let p = count / summary.sample_size;
                let q = 1.0 - p;

                // Thought for later: If q is very close to 0, we could omit this variable from
                // analysis because it's always present and so has no uncertainty.
                bottom_uncertainty -= p * p.log2() + q * q.log2();
                variable
            })
            .collect();

        let variable_count = variables.0.len() as f64;

        ModelEvaluator {
            data,
            sample_size: summary.sample_size,
            variables,
            top: RawEvaluation {
                uncertainty: summary.uncertainty,
                degrees_of_freedom: variable_count.exp2() - 1.0,
            },
            bottom: RawEvaluation {
                uncertainty: bottom_uncertainty,
                degrees_of_freedom: variable_count,
            },
        }
    }

    /// Returns the total count across all cells in the original data.
    ///
    /// Most [`ModelEvaluation`] methods are only meaningful if this is a true sample size, rather
    /// than being 1.0 because the table contains probabilities.
    pub fn sample_size(&self) -> f64 {
        self.sample_size
    }

    /// Returns the set of variables that were observed in the original data.
    pub fn variables(&self) -> &VariableSet<V> {
        &self.variables
    }

    /// Evaluates a model into a [`ModelEvaluation`] which can be used to compare this model to the
    /// saturation and independence models.
    pub fn evaluate<'a>(&'a self, model: &Model<V>) -> ModelEvaluation<'a, V> {
        if model.relations.len() == 1 {
            ModelEvaluation {
                evaluator: self,
                raw: self.top,
            }
        } else if model.relations.iter().all(|relation| relation.0.len() == 1) {
            ModelEvaluation {
                evaluator: self,
                raw: self.bottom,
            }
        } else {
            let plan = model.plan();
            ModelEvaluation {
                evaluator: self,
                raw: RawEvaluation {
                    degrees_of_freedom: plan.degrees_of_freedom(),
                    uncertainty: plan.uncertainty(&self.data),
                },
            }
        }
    }
}

/// Contains the result of evaluating one model against one set of observed data.
///
/// Every method on this type is constant-time (typically just a few floating-point multiplies) and
/// values of this type use only a few words of memory.
pub struct ModelEvaluation<'a, V: VariableId> {
    evaluator: &'a ModelEvaluator<'a, V>,
    raw: RawEvaluation,
}

impl<V: VariableId> ModelEvaluation<'_, V> {
    /// Returns the number of degrees of freedom of this multinomial model.
    pub fn degrees_of_freedom(&self) -> f64 {
        self.raw.degrees_of_freedom
    }

    /// Returns the number of degrees of freedom more that this model has compared to the
    /// independence model.
    pub fn delta_degrees_of_freedom(&self) -> f64 {
        self.raw.degrees_of_freedom - self.evaluator.bottom.degrees_of_freedom
    }

    /// Returns the [Shannon entropy][] for this model, measured in bits.
    ///
    /// [Shannon entropy]: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    ///
    /// Every model from [`Model::less_complex`] will have more entropy than this one.
    pub fn uncertainty(&self) -> f64 {
        self.raw.uncertainty
    }

    /// Returns the number of bits of constraint lost in this model compared to the original data.
    ///
    /// Every model from [`Model::less_complex`] will have more transmission than this one.
    pub fn transmission(&self) -> f64 {
        self.raw.uncertainty - self.evaluator.top.uncertainty
    }

    /// Returns the constraint retained in this model, as a fraction of the constraint lost in the
    /// independence model.
    ///
    /// Every model from [`Model::less_complex`] will have less information than this one.
    pub fn information(&self) -> f64 {
        let ModelEvaluator { bottom, top, .. } = self.evaluator;
        1.0 - self.transmission() / (bottom.uncertainty - top.uncertainty)
    }

    /// Returns the natural logarithm of the [likelihood][] that the observed data was drawn from a
    /// [multinomial distribution][] having the parameters of this model.
    ///
    /// [likelihood]: https://en.wikipedia.org/wiki/Likelihood_function
    /// [multinomial distribution]: https://en.wikipedia.org/wiki/Multinomial_distribution
    ///
    /// Every model from [`Model::less_complex`] will be less likely than this one.
    pub fn log_likelihood(&self) -> f64 {
        self.raw.log_likelihood(self.evaluator.sample_size)
    }

    /// Returns the natural logarithm of the ratio of this model's likelihood compared to the
    /// independence model.
    pub fn delta_log_likelihood(&self) -> f64 {
        self.evaluator
            .bottom
            .log_likelihood(self.evaluator.sample_size)
            - self.log_likelihood()
    }

    /// Returns the difference between the [Akaike Information Criterion][] (AIC) for this model
    /// compared with the independence model. It's desirable to have a lower AIC, but because this
    /// is a difference, higher values are better.
    ///
    /// [Akaike Information Criterion]: https://en.wikipedia.org/wiki/Akaike_information_criterion
    ///
    /// Models from [`Model::less_complex`] may have either higher or lower AIC than this one.
    pub fn akaike_information_criterion(&self) -> f64 {
        let ModelEvaluator {
            sample_size,
            bottom,
            ..
        } = self.evaluator;
        bottom.akaike_information_criterion(*sample_size)
            - self.raw.akaike_information_criterion(*sample_size)
    }

    /// Returns the difference between the [Bayesian Information Criterion][] (BIC) for this model
    /// compared with the independence model. It's desirable to have a lower BIC, but because this
    /// is a difference, higher values are better.
    ///
    /// [Bayesian Information Criterion]: https://en.wikipedia.org/wiki/Bayesian_information_criterion
    ///
    /// Models from [`Model::less_complex`] may have either higher or lower BIC than this one.
    pub fn bayesian_information_criterion(&self) -> f64 {
        let ModelEvaluator {
            sample_size,
            bottom,
            ..
        } = self.evaluator;
        bottom.bayesian_information_criterion(*sample_size)
            - self.raw.bayesian_information_criterion(*sample_size)
    }

    /// Returns the probability that you'd be making a mistake if you claimed that this model has a
    /// different distribution than the one that produced the original data. If you evaluated the
    /// top (most complex) model, this value would be 100%, because in that case the model is
    /// exactly identical to the data.
    pub fn top_alpha(&self) -> f64 {
        self.alpha(self.evaluator.top)
    }

    /// Returns the probability that you'd be making a mistake if you claimed that this model has a
    /// different distribution than the independence model. If you evaluated the bottom (least
    /// complex) model, this value would be 100%, since the models are in fact the same model.
    pub fn bottom_alpha(&self) -> f64 {
        self.alpha(self.evaluator.bottom)
    }

    fn alpha(&self, reference: RawEvaluation) -> f64 {
        let ddf = self.raw.degrees_of_freedom - reference.degrees_of_freedom;
        if ddf == 0.0 {
            return 1.0;
        }
        let lr =
            (reference.log_likelihood(self.evaluator.sample_size) - self.log_likelihood()).abs();
        let chi2 = ChiSquared::new(ddf.abs()).unwrap();
        1.0 - chi2.cdf(2.0 * lr)
    }
}

struct DisjointSets<V>(HashMap<V, (V, u8)>);

impl<V> DisjointSets<V>
where
    V: Copy + Eq + std::hash::Hash,
{
    fn new() -> Self {
        DisjointSets(HashMap::new())
    }

    fn find(&mut self, x: V) -> V {
        self.find_rank(x).0
    }

    fn find_rank(&mut self, mut x: V) -> (V, u8) {
        let mut parent = *self.0.entry(x).or_insert((x, 0));
        while x != parent.0 {
            let grandparent = self.0[&parent.0];
            self.0.insert(x, grandparent);
            x = parent.0;
            parent = grandparent;
        }
        parent
    }

    fn union(&mut self, a: V, b: V) {
        let mut a = self.find_rank(a);
        let mut b = self.find_rank(b);

        if a.0 == b.0 {
            return;
        }

        if a.1 < b.1 {
            swap(&mut a, &mut b);
        }

        self.0.insert(b.0, a);

        if a.1 == b.1 {
            a.1 += 1;
            self.0.insert(a.0, a);
        }
    }
}
