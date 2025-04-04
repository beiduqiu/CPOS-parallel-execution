use crate::binary_tree::BinaryTree;

#[derive(Clone, Debug)]
pub struct MyNode {
    pub id: u32,
    pub label: BinaryTree,
}

impl MyNode {
    pub fn new(id: u32) -> Self {
        MyNode {
            id,
            label: BinaryTree::new(),
        }
    }

    pub fn with_label(id: u32, values: Vec<i32>) -> Self {
        let mut tree = BinaryTree::new();
        for val in values {
            tree.insert(val);
        }
        MyNode { id, label: tree }
    }
}
