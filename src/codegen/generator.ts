import type { Node, Edge } from '@xyflow/react'
import type { LayerNodeData } from '../canvas/GraphStore'

export function generatePyTorchCode(
  _nodes: Node<LayerNodeData>[],
  _edges: Edge[],
): string {
  return `# Phase 2 will turn the graph into a real nn.Module.
# For now this is a placeholder.

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # layers go here

    def forward(self, x):
        return x
`
}
