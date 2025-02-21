mod binary_tree;
mod node;
mod graph_utils;

use petgraph::graph::DiGraph;
use node::MyNode;
use graph_utils::{parse_dot_file_to_digraph, topological_sort_and_execute, visualize_dag};

fn main() {
     // Option 1: Generate a random DAG:
    // let node_count = 6;
    // let edge_count = 8;
    // let mut graph = generate_random_dag(node_count, edge_count);
    // Option 2: Parse the .dot file into a graph of MyNode.
    let dot_file = "dag_out.dot"; // Specify your DOT file path.
    let mut graph: DiGraph<MyNode, ()> = parse_dot_file_to_digraph(dot_file);

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
    use petgraph::graph::DiGraph;
    use crate::node::MyNode;
    use crate::graph_utils::insert_node_label;
    use crate::binary_tree::BinaryTree;

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
