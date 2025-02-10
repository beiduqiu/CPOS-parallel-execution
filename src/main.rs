#[macro_use]
extern crate lazy_static;

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
use std::thread;
use std::time::Duration;
use std::sync::Mutex;

// Global thread registry: maps a node ID to a vector of thread join handles.
lazy_static! {
    static ref GLOBAL_THREAD_REGISTRY: Mutex<HashMap<u32, Vec<thread::JoinHandle<()>>>> = Mutex::new(HashMap::new());
}

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
    
    /// Returns the in‑order traversal of the BST as a vector.
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

/// Custom node type using a BST for its label.
#[derive(Debug, Clone)]
struct MyNode {
    id: u32,
    label: BinaryTree,
}

impl MyNode {
    /// Creates a new MyNode with a label built from the provided vector.
    fn new(label: Vec<i32>) -> Self {
        Self { 
            id: 0, 
            label: BinaryTree::from_vec(label),
        }
    }

    /// Updates the node's label.
    #[allow(dead_code)]
    fn update_label(&mut self, new_label: Vec<i32>) {
        self.label = BinaryTree::from_vec(new_label);
    }
}

/// Inserts a new value into the binary tree label of the node with the given `target_id`.
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
fn generate_random_dag(node_count: usize, edge_count: usize) -> DiGraph<MyNode, ()> {
    let mut graph = DiGraph::<MyNode, ()>::new();
    let mut rng = rand::thread_rng();
    let mut nodes = Vec::new();

    // Create nodes.
    for i in 0..node_count {
        let node = MyNode {
            id: i as u32,
            label: BinaryTree::new(),
        };
        nodes.push(graph.add_node(node));
    }

    // Enforce a topological order.
    let mut topological_order: Vec<_> = (0..node_count).collect();

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
            if petgraph::algo::is_cyclic_directed(&graph) {
                graph.remove_edge(graph.find_edge(src, dst).unwrap());
            } else {
                edges_added += 1;
            }
        }
    }

    // Ensure a single start node (no incoming edges).
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

    // Ensure a single end node (no outgoing edges).
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

    assert!(
        !petgraph::algo::is_cyclic_directed(&graph),
        "Generated graph contains a cycle!"
    );

    graph
}

/// Performs topological sorting on a DiGraph<MyNode, ()> using Kahn's algorithm
/// and spawns threads to simulate node execution.
///
/// For nodes with a single label, a thread is spawned and immediately joined.
/// For nodes with multiple labels:
///   1. The code checks if threads for that node have already been spawned in the global registry.
///   2. If they have been spawned, it directly joins (forks) them.
///   3. If not, it spawns the forked threads (all IDs except the smallest) and records them.
///   4. In both cases, after forking is done, the execution thread (using the smallest thread ID)
///      is spawned and joined.
fn topological_sort_and_execute(graph: &mut DiGraph<MyNode, ()>) -> Vec<MyNode> {
    let mut in_degree: HashMap<NodeIndex, usize> = HashMap::new();
    let mut queue = VecDeque::new();
    let mut sorted_order = Vec::new();
    let mut thread_counter = 0;

    // Initialize in-degrees.
    for node in graph.node_indices() {
        in_degree.insert(node, graph.edges_directed(node, Direction::Incoming).count());
    }

    // Enqueue nodes with zero in-degree.
    let mut num_start_nodes = 0;
    for (&node, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node);
            num_start_nodes += 1;
            // For start nodes, insert 0 into the label.
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
        let node_data = graph[node].clone();
        sorted_order.push(node_data.clone());
        let label_threads = node_data.label.in_order();
        let node_id = node_data.id;

        if label_threads.len() == 1 {
            // For single-threaded nodes: spawn and join immediately.
            let thread_id = label_threads[0];
            let handle = thread::spawn(move || {
                thread::sleep(Duration::from_millis(50));
                println!("Thread {} executes Node {}.", thread_id, node_id);
            });
            {
                let mut registry = GLOBAL_THREAD_REGISTRY.lock().unwrap();
                registry.entry(node_id).or_default().push(handle);
            }
            {
                let mut registry = GLOBAL_THREAD_REGISTRY.lock().unwrap();
                if let Some(handles) = registry.remove(&node_id) {
                    for handle in handles {
                        handle.join().expect("Thread panicked");
                    }
                }
            }
        } else if label_threads.len() > 1 {
            // For multi-threaded nodes, choose the smallest thread ID for execution.
            let min_thread = *label_threads.iter().min().unwrap();
            // Determine forked thread IDs (all except the smallest).
            let forked_threads: Vec<_> = label_threads.iter().copied().filter(|tid| *tid != min_thread).collect();

            // Check if forked threads for this node have already been spawned.
            {
                let mut registry = GLOBAL_THREAD_REGISTRY.lock().unwrap();
                if registry.contains_key(&node_id) {
                    // Threads have been spawned; directly join (fork) them.
                    if let Some(handles) = registry.remove(&node_id) {
                        for handle in handles {
                            handle.join().expect("Forked thread panicked");
                        }
                    }
                } else {
                    // Threads have not been spawned; spawn them and record their handles.
                    let mut handles = Vec::new();
                    for forked in forked_threads {
                        let node_id_clone = node_id;
                        let handle = thread::spawn(move || {
                            thread::sleep(Duration::from_millis(50));
                            println!("Forked thread {} working for Node {}.", forked, node_id_clone);
                        });
                        handles.push(handle);
                    }
                    registry.insert(node_id, handles);
                    // Immediately join the newly spawned forked threads.
                    if let Some(handles) = registry.remove(&node_id) {
                        for handle in handles {
                            handle.join().expect("Forked thread panicked");
                        }
                    }
                }
            }

            // Now spawn the execution thread (using the smallest thread ID) and join it.
            {
                let handle_exec = thread::spawn(move || {
                    thread::sleep(Duration::from_millis(50));
                    println!("Thread {} executes Node {}.", min_thread, node_id);
                });
                let mut registry = GLOBAL_THREAD_REGISTRY.lock().unwrap();
                registry.entry(node_id).or_default().push(handle_exec);
            }
            {
                let mut registry = GLOBAL_THREAD_REGISTRY.lock().unwrap();
                if let Some(handles) = registry.remove(&node_id) {
                    for handle in handles {
                        handle.join().expect("Execution thread panicked");
                    }
                }
            }
        }

        // Process each neighbor.
        let mut neighbors: Vec<_> = graph.neighbors(node).collect();
        neighbors.sort();
        let mut count = 0;
        for neighbor in neighbors {
            if let Some(degree) = in_degree.get_mut(&neighbor) {
                *degree -= 1;
                if count == 0 {
                    let val = graph[node].label.in_order()[0];
                    let _ = insert_node_label(graph, neighbor.index() as u32, val);
                    count += 1;
                } else {
                    thread_counter += 1;
                    let _ = insert_node_label(graph, neighbor.index() as u32, thread_counter);
                }
                if *degree == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
    }
    sorted_order
}

/// Visualizes the DAG by generating a DOT file and converting it to an image using Graphviz.
fn visualize_dag(graph: &DiGraph<MyNode, ()>, dot_file: &str, image_file: &str) {
    let dot_output = format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));
    let mut file = File::create(dot_file).expect("Failed to create DOT file");
    file.write_all(dot_output.as_bytes()).expect("Failed to write DOT file");
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
fn parse_dot_file_to_digraph(file_path: &str) -> DiGraph<MyNode, ()> {
    let dot_content = fs::read_to_string(file_path).expect("Failed to read DOT file");
    let mut graph = DiGraph::<MyNode, ()>::new();
    let mut node_map: HashMap<u32, NodeIndex> = HashMap::new();

    for line in dot_content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("digraph") || line.starts_with("{") || line.starts_with("}") {
            continue;
        }
        if line.contains("->") {
            let parts: Vec<&str> = line.split("->").collect();
            if parts.len() != 2 {
                eprintln!("Warning: Unexpected edge format: {}", line);
                continue;
            }
            let src_str = parts[0].trim();
            let dst_part = parts[1].trim();
            let dst_str = if let Some((id, _)) = dst_part.split_once('[') { id.trim() } else { dst_part };
            let src_id: u32 = src_str.parse().expect("Source node ID should be an integer");
            let dst_id: u32 = dst_str.parse().expect("Destination node ID should be an integer");
            let src_index = *node_map.entry(src_id).or_insert_with(|| {
                graph.add_node(MyNode { id: src_id, label: BinaryTree::new() })
            });
            let dst_index = *node_map.entry(dst_id).or_insert_with(|| {
                graph.add_node(MyNode { id: dst_id, label: BinaryTree::new() })
            });
            graph.add_edge(src_index, dst_index, ());
        } else if line.contains('[') {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }
            let node_str = tokens[0].trim_end_matches(';');
            let node_id: u32 = node_str.parse().expect("Node ID should be an integer");
            let label_vec: Vec<i32> = if let Some(quote_start) = line.find('"') {
                let rest = &line[quote_start + 1..];
                if let Some(quote_end) = rest.find('"') {
                    let label_string = &rest[..quote_end];
                    if let Some(idx) = label_string.find("label:") {
                        let remainder = &label_string[idx + 6..].trim();
                        let remainder = remainder.trim_start_matches('[').trim_end_matches(']');
                        if remainder.is_empty() { Vec::new() } else { remainder.split(',').filter_map(|s| s.trim().parse::<i32>().ok()).collect() }
                    } else { Vec::new() }
                } else { Vec::new() }
            } else { Vec::new() };
            node_map.entry(node_id).or_insert_with(|| {
                graph.add_node(MyNode { id: node_id, label: BinaryTree::from_vec(label_vec) })
            });
        }
    }
    graph
}

fn main() {
    // Option 1: Generate a random DAG:
    // let node_count = 6;
    // let edge_count = 8;
    // let mut graph = generate_random_dag(node_count, edge_count);

    // Option 2: Parse the .dot file into a graph of MyNode.
    let dot_file = "dag_out.dot"; // Specify your DOT file path.
    let mut graph = parse_dot_file_to_digraph(dot_file);

    // Perform topological sort and simulate execution.
    let sorted_order = topological_sort_and_execute(&mut graph);
    println!("Topological Order (node labels in in-order):");
    for node in sorted_order {
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
        let mut graph: DiGraph<MyNode, ()> = DiGraph::new();
        graph.add_node(MyNode { id: 0, label: BinaryTree::new() });
        graph.add_node(MyNode { id: 1, label: BinaryTree::new() });
        graph.add_node(MyNode { id: 2, label: BinaryTree::new() });
        
        for node in graph.node_weights() {
            assert_eq!(node.label.in_order(), Vec::<i32>::new(), "Node {} initial label should be empty", node.id);
        }
        
        let updated = insert_node_label(&mut graph, 1, 42);
        assert!(updated, "The node with id 1 should be updated");
        
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
