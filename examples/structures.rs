use reconstructability::{Model, VariableSet};
use std::collections::HashMap;
use std::fmt;

fn fmt_relation<W: fmt::Write>(out: &mut W, relation: &VariableSet<u8>) -> fmt::Result {
    for variable in relation.iter() {
        out.write_char((b'A' + variable).into())?;
    }
    Ok(())
}

fn fmt_model<W: fmt::Write>(out: &mut W, model: &Model<u8>) -> fmt::Result {
    let mut iter = model.iter();
    if let Some(relation) = iter.next() {
        fmt_relation(out, relation)?;
    }
    for relation in iter {
        out.write_str(":")?;
        fmt_relation(out, relation)?;
    }
    Ok(())
}

fn main() -> fmt::Result {
    let size = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let mut saturated = Model::new();
    saturated.add_relation((0u8..size).into_iter().collect());

    let mut current_layer = vec![(saturated, vec![])];
    let mut next_layer = HashMap::new();

    let mut structures = 0;
    let mut edges = 0;

    println!("graph {{ node [shape=plaintext];");
    while !current_layer.is_empty() {
        for (model, parents) in current_layer.drain(..) {
            structures += 1;

            let mut model_name = String::new();
            fmt_model(&mut model_name, &model)?;

            if !parents.is_empty() {
                print!("{{ ");
                for parent in parents {
                    edges += 1;
                    print!("\"{}\" ", parent);
                }
                print!("}} -- ");
            }
            println!("\"{}\";", model_name);

            for next in model.less_complex() {
                next_layer
                    .entry(next)
                    .or_insert_with(Vec::new)
                    .push(model_name.clone());
            }
        }

        current_layer.extend(next_layer.drain());
        current_layer.sort_unstable();
    }
    println!("}} // {} structures, {} edges", structures, edges);

    Ok(())
}
