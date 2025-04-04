mod binary_tree;
mod node;
mod graph_utils;

use petgraph::graph::DiGraph;
use node::MyNode;
use graph_utils::{parse_dot_file_to_digraph, execute_with_thread_pool, visualize_dag};

fn main() {
    let dot_file = "dag.dot";
    let mut graph: DiGraph<MyNode, ()> = parse_dot_file_to_digraph(dot_file);

    let num_threads = 4; // adjust thread count here
    execute_with_thread_pool(&mut graph, num_threads);

    visualize_dag(&graph, "dag.dot", "dag.png");
}
