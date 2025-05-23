// Copyright 2025 Irreducible Inc.

//! This module provides visualization tools for virtual polynomials used
//! in the evalcheck protocol.
//!
//! Each node in the generated graph corresponds to a specific OracleId.
//!
//! Nodes with a red border represent polynomials that require additional
//! randomization via the sumcheck protocol.
//!
//! Nodes positioned on the left side represent oracles coming from the previous round
//!
//! Duplicated nodes (i.e. repeated evaluations of the same oracle at the
//! same point) are marked in grey and are not recomputed.
//!
//! Each .dot file produced by this module represents a separate round
//! of the greedy_evalcheck procedure.

use std::{cmp::Reverse, fs};

use binius_field::TowerField;
use itertools::chain;
use petgraph::{
	dot::{Config, Dot},
	graph::{DiGraph, NodeIndex},
};

use super::{EvalPoint, EvalPointOracleIdMap, EvalcheckMultilinearClaim};
use crate::oracle::{
	MultilinearOracleSet, MultilinearPolyOracle, MultilinearPolyVariant, OracleId,
};

pub struct GraphBuilder<'a, F: TowerField> {
	graph: DiGraph<Node<F>, &'a str>,
	oracles: &'a mut MultilinearOracleSet<F>,
	visited_claims: EvalPointOracleIdMap<(), F>,
	round: usize,
}

#[derive(Debug)]
struct Node<F: TowerField> {
	name: Option<String>,
	oracle: MultilinearPolyOracle<F>,
	id: usize,
	is_already_visited: bool,
}

impl<'a, F: TowerField> GraphBuilder<'a, F> {
	pub fn new(oracles: &'a mut MultilinearOracleSet<F>, round: usize) -> Self {
		Self {
			graph: DiGraph::new(),
			visited_claims: EvalPointOracleIdMap::new(),
			oracles,
			round,
		}
	}

	pub fn build(&mut self, evalcheck_claims: &[EvalcheckMultilinearClaim<F>]) {
		let mut evalcheck_claims = evalcheck_claims.to_vec();

		evalcheck_claims.sort_unstable_by_key(|claim| Reverse(claim.id));

		for claim in evalcheck_claims {
			self.build_tree(claim.id, claim.eval_point.clone());
		}

		let dot = Dot::with_attr_getters(
			&self.graph,
			&[Config::EdgeNoLabel, Config::NodeNoLabel],
			&|_, _| String::new(),
			&|_, (_, node)| {
				let label = match &node.name {
					Some(name) => format!(
						"{}\n{}({})\nn_vars:{}",
						name,
						node.oracle.type_str(),
						node.id,
						node.oracle.n_vars,
					),
					None => format!(
						"{}({})\nn_vars:{}",
						node.oracle.type_str(),
						node.id,
						node.oracle.n_vars,
					),
				};

				let color = match node.oracle.variant {
					MultilinearPolyVariant::Shifted(_)
					| MultilinearPolyVariant::Packed(_)
					| MultilinearPolyVariant::Composite(_)
						if !node.is_already_visited =>
					{
						"red"
					}
					_ => "black",
				};

				let fillcolor = if node.is_already_visited {
					"grey"
				} else {
					"white"
				};

				format!(
					"label=\"{label}\", color={color}, style=filled, fillcolor={fillcolor}, fontcolor=black"
				)
			},
		);

		let mut dot_output = format!("{dot:?}");
		dot_output = dot_output.replacen("digraph {", "digraph {\n    rankdir=LR;", 1);

		let filename = format!("./evalcheck-{}-round.dot", self.round);
		fs::write(filename, dot_output).expect("Не удалось записать .dot файл");
	}

	fn build_tree(&mut self, id: OracleId, eval_point: EvalPoint<F>) -> NodeIndex {
		let oracle = &self.oracles[id];

		let is_already_visited = if self.visited_claims.get(id, &eval_point).is_some() {
			true
		} else {
			self.visited_claims.insert(id, eval_point.clone(), ());
			false
		};

		let node = Node {
			name: oracle.name.clone(),
			oracle: oracle.clone(),
			id: id.index(),
			is_already_visited,
		};

		let node_index = self.graph.add_node(node);

		if is_already_visited {
			return node_index;
		}

		use MultilinearPolyVariant::{LinearCombination, Projected, Repeating, ZeroPadded};
		match &oracle.variant {
			Repeating {
				id: child_id,
				log_count,
			} => {
				let n_vars = eval_point.len() - log_count;
				let child_eval = eval_point.slice(0..n_vars);
				self.link_child(node_index, *child_id, child_eval);
			}
			Projected(projected) => {
				let (lo, hi) = eval_point.split_at(projected.start_index());
				let new_eval_point = chain!(lo, projected.values(), hi)
					.copied()
					.collect::<Vec<_>>();
				self.link_child(node_index, projected.id(), new_eval_point.into());
			}
			LinearCombination(linear) => {
				let ids = linear.polys().collect::<Vec<_>>();

				for id in ids {
					let child = self.build_tree(id, eval_point.clone());
					self.graph.add_edge(node_index, child, "");
				}
			}
			ZeroPadded(padded) => {
				let kept = chain!(
					&eval_point[..padded.start_index()],
					&eval_point[padded.start_index() + padded.n_pad_vars()..],
				)
				.copied()
				.collect::<Vec<_>>();
				self.link_child(node_index, padded.id(), kept.into());
			}
			_ => {}
		}

		node_index
	}

	fn link_child(&mut self, parent: NodeIndex, child_id: OracleId, eval_point: EvalPoint<F>) {
		let child = self.build_tree(child_id, eval_point);
		self.graph.add_edge(parent, child, "");
	}
}
