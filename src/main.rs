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

/// A binary search tree for storing i32 values in order.
#[derive(Debug, Clone, PartialEq)]
enum BinaryTree {
    Empty,
    Node {
        value: i32,
        left: Box<BinaryTree>,
        right: Box<BinaryTree>,
    },
}

impl BinaryTree {
    /// Creates a new empty BST.
    fn new() -> Self {
        BinaryTree::Empty
    }
    
    /// Inserts a new value into the BST. Values less than the current node go left;
    /// values greater or equal go right.
    fn insert(&mut self, new_value: i32) {
        match self {
            BinaryTree::Empty => {
                *self = BinaryTree::Node {
                    value: new_value,
                    left: Box::new(BinaryTree::Empty),
                    right: Box::new(BinaryTree::Empty),
                }
            },
            BinaryTree::Node { value, left, right } => {
                if new_value < *value {
                    left.insert(new_value)
                } else {
                    right.insert(new_value)
                }
            }
        }
    }
    
    /// Builds a BST from a vector of i32 values by inserting them one by one.
    /// (If the provided vector is empty, the resulting tree will be empty.)
    fn from_vec(mut values: Vec<i32>) -> Self {
        let mut tree = BinaryTree::new();
        for val in values.drain(..) {
            tree.insert(val);
        }
        tree
    }
    
    /// Returns the in‑order traversal of the BST as a vector. In‑order traversal
    /// of a BST produces the values in sorted order.
    fn in_order(&self) -> Vec<i32> {
        match self {
            BinaryTree::Empty => Vec::new(),
            BinaryTree::Node { value, left, right } => {
                let mut result = left.in_order();
                result.push(*value);
                result.extend(right.in_order());
                result
            }
        }
    }
}

/// The custom node structure now uses a BST for its label.
#[derive(Debug, Clone)]
struct MyNode {
    id: u32,
    label: BinaryTree,
}

impl MyNode {
    /// Creates a new MyNode with a label built from the provided vector.
    /// (If the vector is empty, the label will be an empty tree.)
    fn new(label: Vec<i32>) -> Self {
        Self { 
            id: 0, 
            label: BinaryTree::from_vec(label),
        }
    }

    /// Updates the node's label by replacing it with a new BST built from the provided vector.
    #[allow(dead_code)]
    fn update_label(&mut self, new_label: Vec<i32>) {
        self.label = BinaryTree::from_vec(new_label);
    }
}

/// Inserts a new value into the binary tree label of the node with the given `target_id`.
/// Returns `true` if the node was found and the value was inserted, or `false` otherwise.
fn insert_node_label(graph: &mut DiGraph<MyNode, ()>, target_id: u32, new_value: i32) -> bool {
    for node_index in graph.node_indices() {
        if graph[node_index].id == target_id {
            if let Some(node) = graph.node_weight_mut(node_index) {
                node.label.insert(new_value);
                return true;
            }
        }
    }
    false
}

/// Generates a random DAG with MyNode nodes.
/// Each node is created with an id and its label is initialized to an empty BST.
fn generate_random_dag(node_count: usize, edge_count: usize) -> DiGraph<MyNode, ()> {
    let mut graph = DiGraph::<MyNode, ()>::new();
    let mut rng = rand::thread_rng();
    let mut nodes = Vec::new();

    // Create nodes.
    for i in 0..node_count {
        let node = MyNode {
            id: i as u32,
            label: BinaryTree::new(), // Initialize label with an empty tree.
        };
        nodes.push(graph.add_node(node));
    }

    // Enforce a topological order by (optionally) shuffling node indices.
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

            // Remove edge if it creates a cycle.
            if petgraph::algo::is_cyclic_directed(&graph) {
                graph.remove_edge(graph.find_edge(src, dst).unwrap());
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

/// Performs topological sorting on a DiGraph<MyNode, ()> using Kahn's algorithm.
/// Returns a vector of MyNode in topologically sorted order.
fn topological_sort(graph: &mut DiGraph<MyNode, ()>) -> Vec<MyNode> {
    let mut in_degree: HashMap<NodeIndex, usize> = HashMap::new();
    let mut queue = VecDeque::new();
    let mut sorted_order = Vec::new();
    let mut num_start_nodes = 0;

    let mut threadCounter = 0;

    // Initialize in-degrees.
    for node in graph.node_indices() {
        in_degree.insert(node, graph.edges_directed(node, Direction::Incoming).count());
    }

    // Enqueue nodes with zero in-degree.
    for (&node, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node);
            num_start_nodes += 1;
            // Insert 0 into the label for start nodes.
            let updated = insert_node_label(graph, node.index() as u32, 0);
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
        let mut neighbors: Vec<_> = graph.neighbors(node).collect();
        neighbors.sort();
        let mut count = 0;
        for neighbor in neighbors {
            if let Some(degree) = in_degree.get_mut(&neighbor) {
                *degree -= 1;
                if count == 0 {
                    // Insert the next value into the label.
                let updated = insert_node_label(graph, neighbor.index() as u32, graph[node].label.in_order()[0]);
                count += 1;
                 }else{
                    threadCounter += 1;
                    let updated = insert_node_label(graph, neighbor.index() as u32, threadCounter);
                 }
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

/// Visualizes the DAG by generating a DOT file and converting it to an image using Graphviz.
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

/// Parses a DOT file to create a DiGraph<MyNode, ()>.
/// Expects node definitions like: 
///    0 [ label = "MyNode { id: 0, label: [] }" ]
/// and edge definitions like:
///    0 -> 1 [ ]
fn parse_dot_file_to_digraph(file_path: &str) -> DiGraph<MyNode, ()> {
    let dot_content = fs::read_to_string(file_path).expect("Failed to read DOT file");

    let mut graph = DiGraph::<MyNode, ()>::new();
    let mut node_map: HashMap<u32, NodeIndex> = HashMap::new();

    for line in dot_content.lines() {
        let line = line.trim();

        // Skip empty and syntax lines.
        if line.is_empty() 
            || line.starts_with("digraph")
            || line.starts_with("{")
            || line.starts_with("}") {
            continue;
        }

        // Process edge definitions.
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
                graph.add_node(MyNode { 
                    id: src_id, 
                    label: BinaryTree::new() // Initialize with an empty tree.
                })
            });
            let dst_index = *node_map.entry(dst_id).or_insert_with(|| {
                graph.add_node(MyNode { 
                    id: dst_id, 
                    label: BinaryTree::new() // Initialize with an empty tree.
                })
            });

            graph.add_edge(src_index, dst_index, ());
        }
        // Process node definitions.
        else if line.contains('[') {
            // For example: 0 [ label = "MyNode { id: 0, label: [] }" ]
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }
            let node_str = tokens[0].trim_end_matches(';');
            let node_id: u32 = node_str.parse().expect("Node ID should be an integer");

            // Extract the label values from inside the quotes.
            let label_vec: Vec<i32> = if let Some(quote_start) = line.find('"') {
                let rest = &line[quote_start + 1..];
                if let Some(quote_end) = rest.find('"') {
                    let label_string = &rest[..quote_end];
                    if let Some(idx) = label_string.find("label:") {
                        let remainder = &label_string[idx + 6..];
                        let remainder = remainder.trim();
                        let remainder = remainder.trim_start_matches('[').trim_end_matches(']');
                        if remainder.is_empty() {
                            Vec::new()
                        } else {
                            remainder.split(',')
                                .filter_map(|s| s.trim().parse::<i32>().ok())
                                .collect()
                        }
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            node_map.entry(node_id).or_insert_with(|| {
                graph.add_node(MyNode {
                    id: node_id,
                    label: BinaryTree::from_vec(label_vec),
                })
            });
        }
    }

    graph
}

fn main() {
    // Uncomment the following lines to generate a random DAG:
    // let node_count = 6;
    // let edge_count = 8;
    // let graph = generate_random_dag(node_count, edge_count);

    // Parse the .dot file into a graph of MyNode.
    let dot_file = "dag_out.dot"; // Specify your DOT file path.
    let mut graph = parse_dot_file_to_digraph(dot_file);

    // Perform topological sorting.
    let sorted_order = topological_sort(&mut graph);
    println!("Topological Order:");
    for node in sorted_order {
        // We print the in-order traversal of the BST so that the values appear sorted.
        println!("Node {}: label (in-order) {:?}", node.id, node.label.in_order());
    }

    // Visualize the graph.
    visualize_dag(&graph, "dag_out.dot", "dag_out.png");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_insert_node_label() {
        // Create a graph with some nodes.
        let mut graph: DiGraph<MyNode, ()> = DiGraph::new();
        graph.add_node(MyNode { id: 0, label: BinaryTree::new() });
        graph.add_node(MyNode { id: 1, label: BinaryTree::new() });
        graph.add_node(MyNode { id: 2, label: BinaryTree::new() });
        
        // Verify initial labels (via in-order traversal, should be empty).
        for node in graph.node_weights() {
            assert_eq!(node.label.in_order(), Vec::<i32>::new(), "Node {} initial label should be empty", node.id);
        }
        
        // Insert a value for the node with id 1.
        let updated = insert_node_label(&mut graph, 1, 42);
        assert!(updated, "The node with id 1 should be updated");
        
        // Check that node 1's label is updated while the others remain unchanged.
        for node in graph.node_weights() {
            match node.id {
                0 => assert_eq!(node.label.in_order(), Vec::<i32>::new()),
                1 => assert_eq!(node.label.in_order(), vec![42]),
                2 => assert_eq!(node.label.in_order(), Vec::<i32>::new()),
                _ => panic!("Unexpected node id: {}", node.id),
            }
        }
    }
}
