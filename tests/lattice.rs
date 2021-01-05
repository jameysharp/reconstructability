use reconstructability::Model;
use std::collections::{HashMap, HashSet};

macro_rules! check_size {
    ($($name:ident)*) => {
        $(
        #[test]
        fn $name() {
            check(stringify!($name).as_bytes().last().unwrap() - b'0');
        }
        )*
    }
}

check_size!{
    lattice_over_2
    lattice_over_3
    lattice_over_4
    lattice_over_5
    lattice_over_6
    lattice_over_7
    lattice_over_8
    lattice_over_9
}

fn check(variables: u8) {
    let variables = (0..variables).into_iter().collect();
    let independence = Model::independence(&variables);
    let mut saturated = Model::new();
    saturated.add_relation(variables);

    let mut seen = 0;
    let mut current_layer = vec![(saturated, HashSet::new())];
    while !current_layer.is_empty() && seen < 1000 {
        //println!("down from {} models", current_layer.len());
        check_down(&mut current_layer);
        seen += current_layer.len();
    }

    let mut seen = 0;
    current_layer.clear();
    current_layer.push((independence, HashSet::new()));
    while !current_layer.is_empty() && seen < 1000 {
        //println!("up from {} models", current_layer.len());
        check_up(&mut current_layer);
        seen += current_layer.len();
    }
}

fn check_down(current_layer: &mut Vec<(Model<u8>, HashSet<Model<u8>>)>) {
    let mut next_layer = HashMap::new();
    for (model, progenitors) in current_layer.drain(..) {
        let mut computed = HashSet::new();
        for progenitor in model.more_complex() {
            let unique = computed.insert(progenitor);
            // more_complex must not produce duplicates
            assert!(unique);
        }
        assert_eq!(computed, progenitors);

        for next in model.less_complex() {
            let unique = next_layer
                .entry(next)
                .or_insert_with(HashSet::new)
                .insert(model.clone());
            // less_complex must not produce duplicates
            assert!(unique);
        }
    }

    current_layer.extend(next_layer.drain());
}

fn check_up(current_layer: &mut Vec<(Model<u8>, HashSet<Model<u8>>)>) {
    let mut next_layer = HashMap::new();
    for (model, progenitors) in current_layer.drain(..) {
        let mut computed = HashSet::new();
        for progenitor in model.less_complex() {
            let unique = computed.insert(progenitor);
            // less_complex must not produce duplicates
            assert!(unique);
        }
        assert_eq!(computed, progenitors);

        for next in model.more_complex() {
            let unique = next_layer
                .entry(next)
                .or_insert_with(HashSet::new)
                .insert(model.clone());
            // more_complex must not produce duplicates
            assert!(unique);
        }
    }

    current_layer.extend(next_layer.drain());
}
