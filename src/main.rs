use petgraph::dot::{Dot, Config};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{HashSet, HashMap, VecDeque};
use std::fs::File;
use std::io::Write;
use std::process::Command;
use petgraph::dot;
use std::fs;
use std::io;

/// Generate a random DAG with a single start and end node
fn generate_random_dag(node_count: usize, edge_count: usize) -> DiGraph<u32, ()> {
    let mut graph = DiGraph::<u32, ()>::new();
    let mut rng = rand::thread_rng();
    let mut nodes = vec![];

    // Create nodes
    for i in 0..node_count {
        nodes.push(graph.add_node(i as u32));
    }

    // Shuffle nodes to enforce a topological order
    let mut topological_order: Vec<_> = (0..node_count).collect();
    //topological_order.shuffle(1);

    let mut edges_added = 0;
    let mut edges = HashSet::new();

    // Add random edges while ensuring acyclicity
    while edges_added < edge_count {
        let src_idx = rng.gen_range(0..node_count - 1);
        let dst_idx = rng.gen_range(src_idx + 1..node_count); // Ensure src < dst

        let src = nodes[topological_order[src_idx]];
        let dst = nodes[topological_order[dst_idx]];

        if edges.insert((src, dst)) {
            graph.add_edge(src, dst, ());

            // Check if the graph remains a DAG
            if petgraph::algo::is_cyclic_directed(&graph) {
                graph.remove_edge(graph.find_edge(src, dst).unwrap()); // Remove the edge if it creates a cycle
            } else {
                edges_added += 1;
            }
        }
    }

    // Ensure a single start node (no incoming edges)
    let start_node = nodes[0];
    for &node in &nodes[1..] {
        if graph.edges_directed(node, Direction::Incoming).count() == 0 {
            graph.add_edge(start_node, node, ());
        }
        if petgraph::algo::is_cyclic_directed(&graph) {
            graph.remove_edge(graph.find_edge(start_node, node).unwrap()); // Remove the edge if it creates a cycle
        } else {
            edges_added += 1;
        }
    }

    // Ensure a single end node (no outgoing edges)
    let end_node = nodes[node_count - 1];
    for &node in &nodes[..node_count - 1] {
        if graph.edges_directed(node, Direction::Outgoing).count() == 0 && node != end_node {
            graph.add_edge(node, end_node, ());
        }
        if petgraph::algo::is_cyclic_directed(&graph) {
            graph.remove_edge(graph.find_edge(node, end_node).unwrap()); // Remove the edge if it creates a cycle
        } else {
            edges_added += 1;
        }
    }

    // Final check to ensure there are no cycles
    assert!(!petgraph::algo::is_cyclic_directed(&graph), "Generated graph contains a cycle!");

    graph
}

/// Function to perform topological sort using Kahn's Algorithm
fn topological_sort(graph: &DiGraph<u32, ()>) -> Vec<u32> {
    let mut in_degree: HashMap<NodeIndex, usize> = HashMap::new();
    let mut queue = VecDeque::new();
    let mut sorted_order = Vec::new();

    // Initialize in-degree count
    for node in graph.node_indices() {
        in_degree.insert(node, graph.edges_directed(node, Direction::Incoming).count());
    }

    // Find nodes with zero in-degree
    for (&node, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node);
        }
    }

    // Process nodes using Kahn's Algorithm
    while let Some(node) = queue.pop_front() {
        sorted_order.push(graph[node]); // Store the node value
        for neighbor in graph.neighbors(node) {
            if let Some(degree) = in_degree.get_mut(&neighbor) {
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    // Check for cycles
    if sorted_order.len() == graph.node_count() {
        sorted_order
    } else {
        panic!("The generated graph is not a DAG (contains a cycle).");
    }
}

/// Function to visualize the DAG and save it as an image
fn visualize_dag(graph: &DiGraph<u32, ()>, dot_file: &str, image_file: &str) {
    let dot_output = format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));

    // Write DOT file
    let mut file = File::create(dot_file).expect("Failed to create DOT file");
    file.write_all(dot_output.as_bytes()).expect("Failed to write DOT file");
    println!("DOT file saved as: {}", dot_file);

    // Convert DOT to PNG using Graphviz
    let output = Command::new("dot")
        .args(&["-Tpng", dot_file, "-o", image_file])
        .output();

    match output {
        Ok(_) => println!("Graph image saved as: {}", image_file),
        Err(e) => println!("Failed to generate image: {}", e),
    }
}
/// Custom parser to read a `.dot` file and create a `DiGraph<u32, ()>`
fn parse_dot_file_to_digraph(file_path: &str) -> DiGraph<u32, ()> {
    let dot_content = fs::read_to_string(file_path).expect("Failed to read DOT file");

    let mut graph = DiGraph::<u32, ()>::new();
    let mut node_map = HashMap::new();

    for line in dot_content.lines() {
        let line = line.trim();

        // Match nodes
        if line.starts_with(char::is_numeric) && line.contains("[") {
            // Extract node ID
            if let Some((node_id, _)) = line.split_once(" [") {
                let node_id: u32 = node_id.trim().parse().expect("Node ID should be an integer");
                let node_index = graph.add_node(node_id);
                node_map.insert(node_id, node_index);
            }
        }

        // Match edges
        if line.contains("->") {
            // Extract source and destination nodes
            if let Some((src, dst)) = line.split_once("->") {
                let src_id: u32 = src.trim().parse().expect("Source node ID should be an integer");
                let dst_id: u32 = dst.split_once("[").map_or_else(
                    || dst.trim().parse().expect("Destination node ID should be an integer"),
                    |(id, _)| id.trim().parse().expect("Destination node ID should be an integer"),
                );

                // Add edge if both nodes exist
                if let (Some(&src_idx), Some(&dst_idx)) = (node_map.get(&src_id), node_map.get(&dst_id)) {
                    graph.add_edge(src_idx, dst_idx, ());
                }
            }
        }
    }

    graph
}


fn main() {
    // let node_count = 6; // Changeable
    // let edge_count = 8; // Changeable

    // // Generate a random DAG
    // let graph = generate_random_dag(node_count, edge_count);
    let dot_file = "dag.dot"; // Your .dot file
    let graph = parse_dot_file_to_digraph(dot_file);


    // Perform topological sorting
    let sorted_order = topological_sort(&graph);
    println!("Topological Order: {:?}", sorted_order);

    // Visualize the DAG
    visualize_dag(&graph, "dag.dot", "dag.png");
}
