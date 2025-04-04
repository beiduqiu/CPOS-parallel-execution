use petgraph::dot::{Dot, Config};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::fs;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, Condvar};
use std::thread;
use std::time::Duration;

use crate::node::MyNode;
use crate::binary_tree::BinaryTree;

pub fn parse_dot_file_to_digraph(dot_file: &str) -> DiGraph<MyNode, ()> {
    let contents = fs::read_to_string(dot_file).expect("Failed to read .dot file");
    let mut graph = DiGraph::<MyNode, ()>::new();
    let mut node_map = HashMap::new();
    let mut edges = Vec::new();

    // First pass: collect all nodes
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "digraph {" || trimmed == "}" {
            continue;
        }

        if let Some((from, to_with_rest)) = trimmed.split_once("->") {
            let from_id = from.trim().parse::<u32>().unwrap();
            let to_part = to_with_rest.trim();
            let to_id_str = to_part.split_whitespace().next().unwrap().trim_end_matches(';');
            let to_id = to_id_str.parse::<u32>().unwrap();
            edges.push((from_id, to_id));
        }else if let Some((id_str, _)) = trimmed.split_once("[") {
            let id_str = id_str.trim();
            if let Ok(id) = id_str.parse::<u32>() {
                let node = MyNode::new(id);
                let index = graph.add_node(node);
                node_map.insert(id, index);
                println!("Parsed node: {}", id);
            }
        }
    }

    // Second pass: add all edges after all nodes are known
    for (from_id, to_id) in edges {
        if let (Some(&from_index), Some(&to_index)) = (node_map.get(&from_id), node_map.get(&to_id)) {
            graph.add_edge(from_index, to_index, ());
            println!("Parsed edge: {} -> {}", from_id, to_id);
        }
    }

    println!("[INFO] DOT parsing complete. Total nodes: {}, Total edges: {}", graph.node_count(), graph.edge_count());
    graph
}

pub fn visualize_dag(graph: &DiGraph<MyNode, ()>, _dot_file: &str, output_image: &str) {
    let dot = Dot::with_config(&graph, &[Config::EdgeNoLabel]);
    let temp_path = "__graph_temp.dot";
    let mut file = File::create(temp_path).unwrap();
    write!(file, "{:?}", dot).unwrap();

    Command::new("dot")
        .args(["-Tpng", temp_path, "-o", output_image])
        .output()
        .expect("Failed to run dot command");

    let _ = fs::remove_file(temp_path);
}

pub fn execute_with_thread_pool(graph: &mut DiGraph<MyNode, ()>, num_threads: usize) {
    let in_degree = Arc::new(Mutex::new(
        graph
            .node_indices()
            .map(|n| (n, graph.edges_directed(n, Direction::Incoming).count()))
            .collect::<HashMap<_, _>>(),
    ));

    let completed = Arc::new((Mutex::new(HashMap::new()), Condvar::new()));

    let mut thread_jobs: Vec<Vec<NodeIndex>> = vec![vec![]; num_threads];
    let mut levels = compute_levels(graph);
    levels.sort_by_key(|&(_, level)| level);

    for (i, &(node, _)) in levels.iter().enumerate() {
        thread_jobs[i % num_threads].push(node);
    }

    let mut handles = vec![];
    for (tid, jobs) in thread_jobs.into_iter().enumerate() {
        let graph = Arc::new(Mutex::new(graph.clone()));
        let completed = Arc::clone(&completed);

        let handle = thread::spawn(move || {
            for node in jobs {
                loop {
                    let mut completed_map = completed.0.lock().unwrap();
                    let all_parents_done = {
                        let g = graph.lock().unwrap();
                        g.edges_directed(node, Direction::Incoming)
                            .all(|e| completed_map.contains_key(&e.source()))
                    };
                    if all_parents_done {
                        break;
                    }
                    completed_map = completed.1.wait(completed_map).unwrap();
                }

                {
                    let g = graph.lock().unwrap();
                    let node_data = &g[node];
                    println!("Thread {} executing Node {}", tid, node_data.id);
                    thread::sleep(Duration::from_millis(50));
                }

                {
                    let mut completed_map = completed.0.lock().unwrap();
                    completed_map.insert(node, true);
                    completed.1.notify_all();
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

fn compute_levels(graph: &DiGraph<MyNode, ()>) -> Vec<(NodeIndex, usize)> {
    let mut in_degree: HashMap<NodeIndex, usize> = graph
        .node_indices()
        .map(|n| (n, graph.edges_directed(n, Direction::Incoming).count()))
        .collect();
    let mut queue = VecDeque::new();
    let mut levels: HashMap<NodeIndex, usize> = HashMap::new();

    for (&node, &deg) in &in_degree {
        if deg == 0 {
            queue.push_back(node);
            levels.insert(node, 0);
        }
    }

    while let Some(node) = queue.pop_front() {
        let current_level = levels[&node];
        for neighbor in graph.neighbors_directed(node, Direction::Outgoing) {
            in_degree.entry(neighbor).and_modify(|d| *d -= 1);
            if in_degree[&neighbor] == 0 {
                queue.push_back(neighbor);
                levels.insert(neighbor, current_level + 1);
            }
        }
    }

    levels.into_iter().collect()
}
