use reconstructability::Model;
use std::collections::{HashMap, HashSet};

#[test]
fn reversible_lattice() {
    let mut saturated = Model::new();
    saturated.add_relation((0u8..4).into_iter().collect());

    let mut current_layer = vec![(saturated, HashSet::new())];
    let mut next_layer = HashMap::new();

    while !current_layer.is_empty() {
        for (model, parents) in current_layer.drain(..) {
            let computed = model.more_complex().collect::<HashSet<_>>();
            assert_eq!(computed, parents);

            for next in model.less_complex() {
                next_layer
                    .entry(next)
                    .or_insert_with(HashSet::new)
                    .insert(model.clone());
            }
        }

        current_layer.extend(next_layer.drain());
    }
}
