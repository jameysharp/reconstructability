use csv_core::{ReadFieldResult, ReaderBuilder};
use lasso::{Rodeo, RodeoResolver};
use reconstructability::*;
use std::collections::HashSet;
use std::io;
use std::str;

fn load_data<I: io::Read, V: VariableId + Default + lasso::Key>(
    mut input: I,
) -> io::Result<(RodeoResolver<V>, Table<V>)> {
    let mut inputbuf = [0; 16384];
    let mut fieldbuf = [0; 1024];
    let mut fieldlen = 0;
    let mut record = Vec::new();
    let mut count = None;
    let mut table = Table::new();
    let mut rodeo = Rodeo::new();
    let mut tsv = ReaderBuilder::new().delimiter(b'\t').build();

    loop {
        let read = input.read(&mut inputbuf)?;
        let mut bytes = &inputbuf[..read];
        loop {
            let (result, nin, nout) = tsv.read_field(bytes, &mut fieldbuf[fieldlen..]);
            bytes = &bytes[nin..];
            fieldlen += nout;
            match result {
                ReadFieldResult::InputEmpty => break,
                ReadFieldResult::OutputFull => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("variable name too long on line {}", tsv.line()),
                    ));
                }
                ReadFieldResult::Field { record_end } => {
                    let field = str::from_utf8(&fieldbuf[..fieldlen])
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                    fieldlen = 0;

                    let consumed = if count.is_none() {
                        if let Ok(n) = field.parse() {
                            count = Some(n);
                            true
                        } else {
                            count = Some(1.0);
                            false
                        }
                    } else {
                        false
                    };

                    if !consumed {
                        record.push(rodeo.get_or_intern(field));
                    }

                    if record_end {
                        let c = count.unwrap_or(1.0);
                        if c > 0.0 {
                            table.add_cell(VariableSet::new(&record), c);
                        }
                        count = None;
                        record.clear();
                    }
                }
                ReadFieldResult::End => {
                    table.shrink_to_fit();
                    return Ok((rodeo.into_resolver(), table));
                }
            }
        }
    }
}

fn main() -> io::Result<()> {
    let (_resolver, table) = load_data::<_, lasso::MiniSpur>(io::stdin().lock())?;
    let evaluator = ModelEvaluator::new(&table);

    println!("data: {:?}", table);
    println!("  sample size: {}", evaluator.sample_size());

    let mut saturated = Model::new();
    saturated.add_relation(evaluator.variables().clone());

    let mut current_layer = vec![saturated];
    let mut next_layer = HashSet::new();

    loop {
        let model = if let Some(model) = current_layer.pop() {
            for next in model.less_complex() {
                next_layer.insert(next);
            }
            model
        } else {
            current_layer.extend(next_layer.drain());
            if current_layer.is_empty() {
                break;
            }
            current_layer.sort_unstable_by(|a, b| b.cmp(a));
            continue;
        };

        println!();
        println!("{:?}:", model);

        let plan = model.plan();
        println!("  plan: {:?}", plan);

        let evaluation = evaluator.evaluate(&model);
        println!("  dDF: {}", evaluation.delta_degrees_of_freedom());
        println!(
            "  log of likelihood ratio: {:.5}",
            evaluation.delta_log_likelihood()
        );
        println!("  uncertainty: {:.5} bits", evaluation.uncertainty());
        println!("  transmission: {:.5} bits", evaluation.transmission());
        println!("  information: {:.2}%", evaluation.information() * 100.0);
        println!(
            "  Akaike Information Criterion: {:.2}",
            evaluation.akaike_information_criterion()
        );
        println!(
            "  Bayesian Information Criterion: {:.2}",
            evaluation.bayesian_information_criterion()
        );
        println!(
            "  error probability in rejecting equivalence to saturation: {:.2}%",
            evaluation.top_alpha() * 100.0
        );
        println!(
            "  error probability in rejecting equivalence to independence: {:.2}%",
            evaluation.bottom_alpha() * 100.0
        );
    }

    Ok(())
}
