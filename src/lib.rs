use lasso::{Key, MiniSpur};
use smallvec::SmallVec;
use statrs::distribution::{ChiSquared, Univariate};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::f64::consts::LN_2;
use std::iter::FromIterator;
use std::mem::{size_of, swap};

pub type VariableId = MiniSpur;

// SmallVec contains two `usize` fields which overlap with the inline vector, so this size is the
// biggest the inline vector can be without making SmallVec any bigger.
#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VariableSet(SmallVec<[VariableId; 2 * size_of::<usize>() / size_of::<VariableId>()]>);

impl VariableSet {
    pub fn new(ids: &[VariableId]) -> Self {
        let mut v = SmallVec::from_slice(ids);
        v.sort_unstable();
        VariableSet(v)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// ```
    /// use lasso::Rodeo;
    /// use reconstructability::VariableSet;
    ///
    /// let mut rodeo = Rodeo::new();
    /// let a = rodeo.get_or_intern_static("a");
    /// let b = rodeo.get_or_intern_static("b");
    /// let c = rodeo.get_or_intern_static("c");
    ///
    /// let abc = VariableSet::new(&[b, c, a]);
    /// let ab = VariableSet::new(&[a, b]);
    /// let ac = VariableSet::new(&[a, c]);
    /// let bc = VariableSet::new(&[b, c]);
    ///
    /// assert_eq!(abc.intersection(&ab).collect::<Vec<_>>(), vec![a, b]);
    /// assert_eq!(ab.intersection(&abc).collect::<Vec<_>>(), vec![a, b]);
    /// assert_eq!(abc.intersection(&ac).collect::<Vec<_>>(), vec![a, c]);
    /// assert_eq!(ac.intersection(&abc).collect::<Vec<_>>(), vec![a, c]);
    /// assert_eq!(abc.intersection(&bc).collect::<Vec<_>>(), vec![b, c]);
    /// assert_eq!(bc.intersection(&abc).collect::<Vec<_>>(), vec![b, c]);
    /// assert_eq!(ab.intersection(&ac).collect::<Vec<_>>(), vec![a]);
    /// assert_eq!(ac.intersection(&ab).collect::<Vec<_>>(), vec![a]);
    /// assert_eq!(ab.intersection(&bc).collect::<Vec<_>>(), vec![b]);
    /// assert_eq!(bc.intersection(&ab).collect::<Vec<_>>(), vec![b]);
    /// assert_eq!(ac.intersection(&bc).collect::<Vec<_>>(), vec![c]);
    /// assert_eq!(bc.intersection(&ac).collect::<Vec<_>>(), vec![c]);
    /// ```
    pub fn intersection<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = VariableId> + 'a {
        let mut ys = other.0.iter().copied();
        let mut maybe_y = ys.next();
        self.0.iter().copied().filter(move |x| {
            while let Some(y) = maybe_y {
                if y > *x {
                    break;
                }
                maybe_y = ys.next();
                if y == *x {
                    return true;
                }
            }
            false
        })
    }

    /// ```
    /// use lasso::Rodeo;
    /// use reconstructability::VariableSet;
    ///
    /// let mut rodeo = Rodeo::new();
    /// let a = rodeo.get_or_intern_static("a");
    /// let b = rodeo.get_or_intern_static("b");
    /// let c = rodeo.get_or_intern_static("c");
    ///
    /// let all = vec![a, b, c];
    ///
    /// let abc = VariableSet::new(&[b, c, a]);
    /// let ab = VariableSet::new(&[a, b]);
    /// let ac = VariableSet::new(&[a, c]);
    /// let bc = VariableSet::new(&[b, c]);
    /// let a = VariableSet::new(&[a]);
    /// let b = VariableSet::new(&[b]);
    /// let c = VariableSet::new(&[c]);
    ///
    /// let subsets = [
    ///     (&abc, &ab), (&abc, &ac), (&abc, &bc),
    ///     (&ab, &ac), (&ab, &bc), (&ac, &bc),
    ///     (&a, &bc), (&b, &ac), (&c, &ab),
    /// ];
    ///
    /// for (x, y) in &subsets {
    ///     assert_eq!(&x.union(y).collect::<Vec<_>>(), &all);
    ///     assert_eq!(&y.union(x).collect::<Vec<_>>(), &all);
    /// }
    /// ```
    pub fn union<'a>(&'a self, other: &'a Self) -> impl Iterator<Item = VariableId> + 'a {
        let mut xs = self.0.iter().copied().peekable();
        let mut ys = other.0.iter().copied().peekable();
        std::iter::from_fn(move || match (xs.peek().copied(), ys.peek().copied()) {
            (None, None) => None,
            (Some(_), None) => xs.next(),
            (None, Some(_)) => ys.next(),
            (Some(x), Some(y)) => match x.cmp(&y) {
                Ordering::Less => xs.next(),
                Ordering::Greater => ys.next(),
                Ordering::Equal => {
                    ys.next();
                    xs.next()
                }
            },
        })
    }

    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.intersection(other).count() == self.len()
    }
}

impl std::fmt::Debug for VariableSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set()
            .entries(self.0.iter().map(|v| unsafe { v.into_usize() }))
            .finish()
    }
}

impl FromIterator<VariableId> for VariableSet {
    fn from_iter<I: IntoIterator<Item = VariableId>>(iter: I) -> Self {
        let mut v = SmallVec::from_iter(iter);
        v.sort();
        VariableSet(v)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    pub raw_data: HashMap<VariableSet, f64>,
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct TableSummary {
    pub uncertainty: f64,
    pub sample_size: f64,
}

impl Table {
    pub fn new() -> Self {
        Table {
            raw_data: HashMap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Table {
            raw_data: HashMap::with_capacity(capacity),
        }
    }

    pub fn add_cell(&mut self, state: VariableSet, count: f64) -> &mut Self {
        if count > 0.0 {
            *self.raw_data.entry(state).or_insert(0.0) += count;
        }
        self
    }

    /// ```
    /// use lasso::Rodeo;
    /// use reconstructability::{Table, VariableSet};
    ///
    /// let mut rodeo = Rodeo::new();
    /// let a = rodeo.get_or_intern_static("a");
    /// let b = rodeo.get_or_intern_static("b");
    ///
    /// let ab = VariableSet::new(&[a, b]);
    /// let a = VariableSet::new(&[a]);
    /// let b = VariableSet::new(&[b]);
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
    pub fn project(&self, on: &VariableSet) -> Table {
        // In theory, there are 2**on.len() states in this projected table, so we can preallocate
        // storage for them. But don't preallocate too much capacity; the input data is typically
        // sparse so it may be that many of the possible projected states are empty. Worse, the number
        // of possible states may far exceed available memory, so we can't afford to be over-eager.
        let mut projection = Table::with_capacity((1 << on.len().min(8)).max(self.raw_data.len()));

        // Reuse the same heap allocation for every projection to avoid hammering the allocator.
        let mut projected_state = Vec::with_capacity(on.len());
        for (state, count) in self.raw_data.iter() {
            projected_state.extend(state.intersection(on));
            projection.add_cell(VariableSet::new(&projected_state), *count);
            projected_state.clear();
        }
        projection
    }

    /// ```
    /// use lasso::Rodeo;
    /// use reconstructability::{Table, VariableSet};
    ///
    /// let mut rodeo = Rodeo::new();
    /// let a = rodeo.get_or_intern_static("a");
    /// let b = rodeo.get_or_intern_static("b");
    ///
    /// let ab = VariableSet::new(&[a, b]);
    /// let a = VariableSet::new(&[a]);
    /// let b = VariableSet::new(&[b]);
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
    pub fn compose_with(&self, other: &Self) -> Table {
        let mut composition = Table::with_capacity(self.raw_data.len() * other.raw_data.len());
        let mut sample_size = 0.0;

        // Reuse the same heap allocation for every projection to avoid hammering the allocator.
        let mut composed_state = Vec::new();
        for (state_a, count_a) in self.raw_data.iter() {
            sample_size += count_a;
            for (state_b, count_b) in other.raw_data.iter() {
                composed_state.extend(state_a.union(state_b));
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

    /// ```
    /// use lasso::Rodeo;
    /// use reconstructability::{Table, VariableSet};
    ///
    /// let mut rodeo = Rodeo::new();
    /// let a = rodeo.get_or_intern_static("a");
    /// let b = rodeo.get_or_intern_static("b");
    ///
    /// let mut builder = Table::new();
    /// assert_eq!(builder.summary().uncertainty, 0.0);
    ///
    /// builder.add_cell(VariableSet::new(&[]), 1.0);
    /// assert_eq!(builder.summary().uncertainty, 0.0);
    ///
    /// builder.add_cell(VariableSet::new(&[a]), 1.0);
    /// assert_eq!(builder.summary().uncertainty, 1.0);
    ///
    /// builder.add_cell(VariableSet::new(&[b]), 1.0);
    /// builder.add_cell(VariableSet::new(&[a, b]), 1.0);
    /// assert_eq!(builder.summary().uncertainty, 2.0);
    /// ```
    pub fn summary(&self) -> TableSummary {
        let mut summary = TableSummary {
            uncertainty: 0.0,
            sample_size: 0.0,
        };
        for count in self.raw_data.values() {
            summary.uncertainty -= count * count.log2();
            summary.sample_size += count;
        }
        if summary.sample_size > 0.0 {
            summary.uncertainty /= summary.sample_size;
            summary.uncertainty += summary.sample_size.log2();
        }
        summary
    }

    pub fn shrink_to_fit(&mut self) {
        self.raw_data.shrink_to_fit();
    }
}

/// A model is a set of variable-sets, where no variable-set is a subset of any other in the model.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Model {
    relations: Vec<VariableSet>,
}

impl Model {
    pub fn new() -> Model {
        Model {
            relations: Vec::new(),
        }
    }

    pub fn independence(variables: &VariableSet) -> Model {
        Model {
            relations: variables
                .0
                .iter()
                .map(|variable| VariableSet::new(std::slice::from_ref(variable)))
                .collect(),
        }
    }

    // This implementation relies on the relations being sorted in descending order by number of
    // variables. Break ties by VariableSet's natural order so that there's a canonical order and
    // derived Eq/Ord/Hash just work.
    fn sort_by(a: &VariableSet, b: &VariableSet) -> Ordering {
        b.len().cmp(&a.len()).then_with(|| a.cmp(b))
    }

    /// ```
    /// use lasso::Rodeo;
    /// use reconstructability::{Model, VariableSet};
    ///
    /// let mut rodeo = Rodeo::new();
    /// let a = rodeo.get_or_intern_static("a");
    /// let b = rodeo.get_or_intern_static("b");
    ///
    /// let ab = VariableSet::new(&[a, b]);
    /// let a = VariableSet::new(&[a]);
    /// let b = VariableSet::new(&[b]);
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
    pub fn add_relation(&mut self, relation: VariableSet) -> &mut Model {
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
                .any(|r| relation.is_subset_of(r));
            if !is_subset && len > 0 {
                // If there are any smaller relations subsumed by this one, keep everything except
                // those relations.
                self.relations
                    .retain(|r| len <= r.len() || !r.is_subset_of(&relation));

                // We retained everything of this relation's size or bigger, so at least up to the
                // point of insertion nothing has changed and the position we found with binary
                // search is still valid.
                self.relations.insert(insert_at, relation);
            }
        }
        self
    }

    /// ```
    /// use lasso::Rodeo;
    /// use reconstructability::{Model, VariableSet};
    ///
    /// let mut rodeo = Rodeo::new();
    /// let a = rodeo.get_or_intern_static("a");
    /// let b = rodeo.get_or_intern_static("b");
    /// let c = rodeo.get_or_intern_static("c");
    ///
    /// let mut abc = Model::new();
    /// abc.add_relation(VariableSet::new(&[a, b, c]));
    ///
    /// let ab = VariableSet::new(&[a, b]);
    /// let ac = VariableSet::new(&[a, c]);
    /// let bc = VariableSet::new(&[b, c]);
    /// let a = VariableSet::new(&[a]);
    /// let b = VariableSet::new(&[b]);
    /// let c = VariableSet::new(&[c]);
    ///
    /// let mut df1 = abc.less_complex();
    /// let mut ab_ac_bc = Model::new();
    /// ab_ac_bc.add_relation(ab.clone()).add_relation(ac.clone()).add_relation(bc.clone());
    ///
    /// assert_eq!(df1.next(), Some(ab_ac_bc));
    /// assert_eq!(df1.next(), None);
    /// ```
    pub fn less_complex(&self) -> impl Iterator<Item = Model> + '_ {
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

            // Now add every subset of this relation that is one variable smaller. Start by setting
            // aside the last variable. Working from the end backward produces the subsets in
            // lexicographic order.
            let (withheld, reduced) = removed.0.split_last().unwrap();
            let mut withheld = *withheld;
            let mut reduced = SmallVec::from(reduced);

            for remove_var in (0..reduced.len()).rev() {
                model.add_relation(VariableSet(reduced.clone()));

                // Put the withheld variable back and remove the variable immediately before it.
                // The remove/insert pair doesn't shift any other variables around and this
                // maintains VariableSet's invariant that the variables are in sorted order.
                swap(&mut withheld, &mut reduced[remove_var]);
            }
            model.add_relation(VariableSet(reduced));

            model
        })
    }

    pub fn plan(&self) -> QueryPlan {
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
                                if k.is_subset_of(&relation) {
                                    plan.append(v);
                                    plan.push((k.clone(), 1));
                                    return false;
                                }
                            }
                            Ordering::Greater => {
                                if relation.is_subset_of(k) {
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
        let mut disjoint_loop_vars = DisjointSets::default();
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

#[derive(Clone, Debug)]
pub struct QueryPlan {
    loops: Vec<Vec<VariableSet>>,
    loopless: Vec<(VariableSet, usize)>,
}

impl QueryPlan {
    pub fn degrees_of_freedom(&self) -> f64 {
        let mut total = 0.0;

        for group in self.loops.iter() {
            let mut relations = group.iter();
            while let Some(a) = relations.next() {
                total += (a.0.len() as f64).exp2() - 1.0;
                for b in relations.clone() {
                    total -= (a.intersection(b).count() as f64).exp2() - 1.0;
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

    pub fn uncertainty(&self, table: &Table) -> f64 {
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

    pub fn has_loops(&self) -> bool {
        !self.loops.is_empty()
    }

    pub fn loopless_factors(&self) -> impl Iterator<Item = (&VariableSet, f64)> {
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

fn iterative_proportional_fit(
    projections: &[(&VariableSet, Table)],
    max_iter: u8,
    max_error: f64,
) -> Table {
    fn init<'a>(
        result: &mut Vec<(Vec<&'a VariableSet>, f64)>,
        keys: &mut Vec<&'a VariableSet>,
        margins: &'a [(&'a VariableSet, Table)],
        seen: &VariableSet,
        state: VariableSet,
    ) {
        if let Some(((relation, margin), rest)) = margins.split_first() {
            let common: VariableSet = seen.intersection(relation).collect();
            let seen: VariableSet = seen.union(relation).collect();

            for current in margin.raw_data.keys() {
                if common.intersection(current).eq(common.intersection(&state)) {
                    keys.push(current);
                    init(result, keys, rest, &seen, state.union(current).collect());
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
            let mut state = VariableSet::new(&[]);
            for key in keys {
                state = state.union(key).collect();
            }
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

pub struct ModelEvaluator<'a> {
    data: &'a Table,
    sample_size: f64,
    top: RawEvaluation,
    bottom: RawEvaluation,
}

impl ModelEvaluator<'_> {
    pub fn new<'a>(data: &'a Table, variables: &VariableSet) -> ModelEvaluator<'a> {
        let summary = data.summary();

        let bottom = Model::independence(variables);
        let bottom_plan = bottom.plan();

        ModelEvaluator {
            data,
            sample_size: summary.sample_size,
            top: RawEvaluation {
                uncertainty: summary.uncertainty,
                degrees_of_freedom: (variables.0.len() as f64).exp2() - 1.0,
            },
            bottom: RawEvaluation {
                uncertainty: bottom_plan.uncertainty(&data),
                degrees_of_freedom: variables.0.len() as f64,
            },
        }
    }

    pub fn sample_size(&self) -> f64 {
        self.sample_size
    }

    pub fn evaluate<'a>(&'a self, model: &Model) -> ModelEvaluation<'a> {
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

pub struct ModelEvaluation<'a> {
    evaluator: &'a ModelEvaluator<'a>,
    raw: RawEvaluation,
}

impl ModelEvaluation<'_> {
    pub fn degrees_of_freedom(&self) -> f64 {
        self.raw.degrees_of_freedom
    }

    pub fn delta_degrees_of_freedom(&self) -> f64 {
        self.raw.degrees_of_freedom - self.evaluator.bottom.degrees_of_freedom
    }

    pub fn uncertainty(&self) -> f64 {
        self.raw.uncertainty
    }

    pub fn transmission(&self) -> f64 {
        self.raw.uncertainty - self.evaluator.top.uncertainty
    }

    pub fn information(&self) -> f64 {
        let ModelEvaluator { bottom, top, .. } = self.evaluator;
        1.0 - self.transmission() / (bottom.uncertainty - top.uncertainty)
    }

    pub fn log_likelihood(&self) -> f64 {
        self.raw.log_likelihood(self.evaluator.sample_size)
    }

    pub fn akaike_information_criterion(&self) -> f64 {
        let ModelEvaluator { sample_size, bottom, .. } = self.evaluator;
        bottom.akaike_information_criterion(*sample_size) - self.raw.akaike_information_criterion(*sample_size)
    }

    pub fn bayesian_information_criterion(&self) -> f64 {
        let ModelEvaluator { sample_size, bottom, .. } = self.evaluator;
        bottom.bayesian_information_criterion(*sample_size) - self.raw.bayesian_information_criterion(*sample_size)
    }

    pub fn top_alpha(&self) -> f64 {
        self.alpha(self.evaluator.top)
    }

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

#[derive(Default)]
struct DisjointSets(HashMap<VariableId, (VariableId, u8)>);

impl DisjointSets {
    fn find(&mut self, x: VariableId) -> VariableId {
        self.find_rank(x).0
    }

    fn find_rank(&mut self, mut x: VariableId) -> (VariableId, u8) {
        let mut parent = *self.0.entry(x).or_insert((x, 0));
        while x != parent.0 {
            let grandparent = self.0[&parent.0];
            self.0.insert(x, grandparent);
            x = parent.0;
            parent = grandparent;
        }
        parent
    }

    fn union(&mut self, a: VariableId, b: VariableId) {
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn variableset_size() {
        assert_eq!(size_of::<VariableSet>(), size_of::<SmallVec<[(); 0]>>());
    }
}
