use crate::binary_tree::BinaryTree;

#[derive(Debug, Clone)]
pub struct MyNode {
    pub id: u32,
    pub label: BinaryTree,
}

impl MyNode {
    /// Creates a new MyNode with a label built from the provided vector.
    pub fn new(label: Vec<i32>) -> Self {
        Self { 
            id: 0, 
            label: BinaryTree::from_vec(label),
        }
    }

    /// Updates the node's label.
    pub fn update_label(&mut self, new_label: Vec<i32>) {
        self.label = BinaryTree::from_vec(new_label);
    }
}
