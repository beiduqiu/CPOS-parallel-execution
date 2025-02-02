use petgraph::dot::{Dot, Config};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use rand::Rng;
use std::collections::{HashSet, HashMap, VecDeque};
use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::fs;
use std::io;

// Define your custom node structure with a u32 label.
#[derive(Debug, Clone)]
struct MyNode {
    id: u32,
    label: u32,
}

/// Generate a random DAG with MyNode nodes.
/// Each node is created with id and label set to the same value.
fn generate_random_dag(node_count: usize, edge_count: usize) -> DiGraph<MyNode, ()> {
    let mut graph = DiGraph::<MyNode, ()>::new();
    let mut rng = rand::thread_rng();
    let mut nodes = Vec::new();

    // Create nodes as MyNode.
    for i in 0..node_count {
        let node = MyNode {
            id: i as u32,
            label: i as u32, // Here we simply set the label equal to the id.
        };
        nodes.push(graph.add_node(node));
    }

    // Enforce a topological order by shuffling node indices.
    let mut topological_order: Vec<_> = (0..node_count).collect();
    // Uncomment the following line to randomize the order:
    // topological_order.shuffle(&mut rng);

    let mut edges_added = 0;
    let mut edges = HashSet::new();

    // Add random edges while ensuring acyclicity.
    while edges_added < edge_count {
        let src_idx = rng.gen_range(0..node_count - 1);
        let dst_idx = rng.gen_range(src_idx + 1..node_count); // Ensure src < dst

        let src = nodes[topological_order[src_idx]];
        let dst = nodes[topological_order[dst_idx]];

        if edges.insert((src, dst)) {
            graph.add_edge(src, dst, ());

            // Check if the graph remains a DAG.
            if petgraph::algo::is_cyclic_directed(&graph) {
                graph.remove_edge(graph.find_edge(src, dst).unwrap()); // Remove edge if it creates a cycle.
            } else {
                edges_added += 1;
            }
        }
    }

    // Ensure a single start node (with no incoming edges).
    let start_node = nodes[0];
    for &node in &nodes[1..] {
        if graph.edges_directed(node, Direction::Incoming).count() == 0 {
            graph.add_edge(start_node, node, ());
        }
        if petgraph::algo::is_cyclic_directed(&graph) {
            graph.remove_edge(graph.find_edge(start_node, node).unwrap());
        } else {
            edges_added += 1;
        }
    }

    // Ensure a single end node (with no outgoing edges).
    let end_node = nodes[node_count - 1];
    for &node in &nodes[..node_count - 1] {
        if graph.edges_directed(node, Direction::Outgoing).count() == 0 && node != end_node {
            graph.add_edge(node, end_node, ());
        }
        if petgraph::algo::is_cyclic_directed(&graph) {
            graph.remove_edge(graph.find_edge(node, end_node).unwrap());
        } else {
            edges_added += 1;
        }
    }

    // Final check to ensure there are no cycles.
    assert!(
        !petgraph::algo::is_cyclic_directed(&graph),
        "Generated graph contains a cycle!"
    );

    graph
}

/// Updates the label of the node with the given `target_id` to `new_label`.
/// Returns `true` if the node was found and updated, or `false` otherwise.
fn update_node_label(graph: &mut DiGraph<MyNode, ()>, target_id: u32, new_label: u32) -> bool {
    for node_index in graph.node_indices() {
        if graph[node_index].id == target_id {
            if let Some(node) = graph.node_weight_mut(node_index) {
                node.label = new_label;
                return true;
            }
        }
    }
    false
}

/// Perform topological sorting on a DiGraph<MyNode, ()> using Kahn's algorithm.
/// Returns a vector of MyNode in topologically sorted order.
///
/// Note: We require a mutable borrow here because we call `update_node_label`
/// which requires `&mut graph`.
fn topological_sort(graph: &mut DiGraph<MyNode, ()>) -> Vec<MyNode> {
    let mut in_degree: HashMap<NodeIndex, usize> = HashMap::new();
    let mut queue = VecDeque::new();
    let mut sorted_order = Vec::new();
    let mut num_start_nodes = 0;

    // Initialize in-degree for each node.
    for node in graph.node_indices() {
        in_degree.insert(node, graph.edges_directed(node, Direction::Incoming).count());
    }

    // Find nodes with zero in-degree.
    // (We iterate over our in_degree HashMap, which is independent of the graph borrow.)
    for (&node, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node);
            num_start_nodes += 1;
            // Convert node.index() (usize) to u32.
            let updated = update_node_label(graph, node.index() as u32, 0);
            assert!(updated, "The node with id {} should be updated", node.index());
        }
    }
    if num_start_nodes != 1 {
        println!("Number of start nodes: {}", num_start_nodes);
        panic!("The generated graph does not have a single start node.");
    }

    // Process nodes in topological order.
    while let Some(node) = queue.pop_front() {
        sorted_order.push(graph[node].clone());
        for neighbor in graph.neighbors(node) {
            if let Some(degree) = in_degree.get_mut(&neighbor) {
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    if sorted_order.len() == graph.node_count() {
        sorted_order
    } else {
        panic!("The generated graph is not a DAG (contains a cycle).");
    }
}

/// Visualize the DAG by generating a DOT file and converting it to an image using Graphviz.
fn visualize_dag(graph: &DiGraph<MyNode, ()>, dot_file: &str, image_file: &str) {
    let dot_output = format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));

    let mut file = File::create(dot_file).expect("Failed to create DOT file");
    file.write_all(dot_output.as_bytes())
        .expect("Failed to write DOT file");
    println!("DOT file saved as: {}", dot_file);

    let output = Command::new("dot")
        .args(&["-Tpng", dot_file, "-o", image_file])
        .output();

    match output {
        Ok(_) => println!("Graph image saved as: {}", image_file),
        Err(e) => println!("Failed to generate image: {}", e),
    }
}

/// Parse a .dot file to create a DiGraph<MyNode, ()>.
/// Expects node definitions like: 
///    0 [ label = "MyNode { id: 0, label: 0 }" ]
/// and edge definitions like:
///    0 -> 1 [ ]
fn parse_dot_file_to_digraph(file_path: &str) -> DiGraph<MyNode, ()> {
    let dot_content = fs::read_to_string(file_path).expect("Failed to read DOT file");

    let mut graph = DiGraph::<MyNode, ()>::new();
    let mut node_map: HashMap<u32, NodeIndex> = HashMap::new();

    for line in dot_content.lines() {
        let line = line.trim();

        // Skip empty lines and DOT syntax lines.
        if line.is_empty() 
            || line.starts_with("digraph")
            || line.starts_with("{")
            || line.starts_with("}") {
            continue;
        }

        // Handle edge definitions.
        if line.contains("->") {
            let parts: Vec<&str> = line.split("->").collect();
            if parts.len() != 2 {
                eprintln!("Warning: Unexpected edge format: {}", line);
                continue;
            }
            let src_str = parts[0].trim();
            let dst_part = parts[1].trim();
            let dst_str = if let Some((id, _)) = dst_part.split_once('[') {
                id.trim()
            } else {
                dst_part
            };

            let src_id: u32 = src_str.parse().expect("Source node ID should be an integer");
            let dst_id: u32 = dst_str.parse().expect("Destination node ID should be an integer");

            let src_index = *node_map.entry(src_id).or_insert_with(|| {
                graph.add_node(MyNode { id: src_id, label: src_id })
            });
            let dst_index = *node_map.entry(dst_id).or_insert_with(|| {
                graph.add_node(MyNode { id: dst_id, label: dst_id })
            });

            graph.add_edge(src_index, dst_index, ());
        }
        // Handle node definitions.
        else if line.contains('[') {
            // For example: 0 [ label = "MyNode { id: 0, label: 0 }" ]
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }
            let node_str = tokens[0].trim_end_matches(';');
            let node_id: u32 = node_str.parse().expect("Node ID should be an integer");

            // Extract the label from inside the quotes.
            // The label is expected to be in the format: MyNode { id: X, label: Y }
            let label: u32 = if let Some(quote_start) = line.find('"') {
                let rest = &line[quote_start + 1..];
                if let Some(quote_end) = rest.find('"') {
                    let label_string = &rest[..quote_end];
                    if let Some(idx) = label_string.find("label:") {
                        let remainder = &label_string[idx + 6..];
                        let remainder = remainder.trim();
                        // Take consecutive digits from the start of the remainder.
                        let digits: String = remainder.chars()
                            .take_while(|c| c.is_ascii_digit())
                            .collect();
                        digits.parse::<u32>().unwrap_or(node_id)
                    } else {
                        node_id
                    }
                } else {
                    node_id
                }
            } else {
                node_id
            };

            node_map.entry(node_id).or_insert_with(|| {
                graph.add_node(MyNode {
                    id: node_id,
                    label, // Use the extracted label.
                })
            });
        }
    }

    graph
}

fn main() {
    // Uncomment to generate a random DAG:
    // let node_count = 6;
    // let edge_count = 8;
    // let graph = generate_random_dag(node_count, edge_count);

    // Parse the .dot file into a graph of MyNode.
    let dot_file = "dag_out.dot"; // Your .dot file path.
    let mut graph = parse_dot_file_to_digraph(dot_file);

    // Perform topological sorting.
    let sorted_order = topological_sort(&mut graph);
    println!("Topological Order:");
    for node in sorted_order {
        println!("Node {}: label {}", node.id, node.label);
    }

    // Visualize the graph.
    visualize_dag(&graph, "dag_out.dot", "dag_out.png");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_update_node_label() {
        // Create a graph with some nodes.
        let mut graph: DiGraph<MyNode, ()> = DiGraph::new();
        graph.add_node(MyNode { id: 0, label: 0 });
        graph.add_node(MyNode { id: 1, label: 1 });
        graph.add_node(MyNode { id: 2, label: 2 });
        
        // Verify initial labels.
        for node in graph.node_weights() {
            match node.id {
                0 => assert_eq!(node.label, 0),
                1 => assert_eq!(node.label, 1),
                2 => assert_eq!(node.label, 2),
                _ => panic!("Unexpected node id: {}", node.id),
            }
        }
        
        // Update the label for the node with id 1.
        let updated = update_node_label(&mut graph, 1, 42);
        assert!(updated, "The node with id 1 should be updated");
        
        // Check that node 1's label is updated while the others remain unchanged.
        for node in graph.node_weights() {
            match node.id {
                0 => assert_eq!(node.label, 0),
                1 => assert_eq!(node.label, 42),
                2 => assert_eq!(node.label, 2),
                _ => panic!("Unexpected node id: {}", node.id),
            }
        }
    }
}
