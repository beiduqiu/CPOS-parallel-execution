#[derive(Debug, Clone, PartialEq)]
pub enum BinaryTree {
    Empty,
    Node {
        value: i32,
        left: Box<BinaryTree>,
        right: Box<BinaryTree>,
    },
}

impl BinaryTree {
    pub fn new() -> Self {
        BinaryTree::Empty
    }
    
    pub fn insert(&mut self, new_value: i32) {
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
    
    pub fn from_vec(mut values: Vec<i32>) -> Self {
        let mut tree = BinaryTree::new();
        for val in values.drain(..) {
            tree.insert(val);
        }
        tree
    }
    
    pub fn in_order(&self) -> Vec<i32> {
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
