/// High-level model for binary Merkle trees using the Grøstl-256 output transformation as a 2-to-1
/// compression function.

mod model {
	use binius_m3::builder::{Col, B1, B32, B8};

	pub struct MerkleTree {
        // The root the the leafs belong to
		root_ids: Col<B8>,

        // The hash of [ left || right ]
		parent_data: Col<B8, 64>,

        // Data contained in the left and right children nodes of the parent node
		left_data: Col<B8, 64>,
		right_data: Col<B8, 64>,

        // Depth of the parent node
		parent_depth: Col<B8, 64>,
        // The index of the parent node
		parent_index: Col<B32>,

        // Whether to pull the left or right child
		pull_left: Col<B1>,
		pull_right: Col<B1>,
	}
}
