import torch
import numpy as np
import plotly.graph_objects as go

try:
    from torch_geometric.explain import Explainer, GNNExplainer
    HAS_EXPLAINER = True
except ImportError:
    HAS_EXPLAINER = False

def run_gcn_explainer(model, data):
    """
    Runs GNNExplainer on the given model and graph data.
    Returns node_mask and edge_mask probabilities.
    """
    if not HAS_EXPLAINER:
        return None, None
        
    model.eval()
    
    # Configure Explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )
    
    # We need a batch vector since the model expects batch
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
    
    # Run explainer
    explanation = explainer(data.x, data.edge_index, batch=batch)
    
    # Extract importances
    node_mask = explanation.node_mask.mean(dim=1).detach().cpu().numpy() # Aggregate features
    edge_mask = explanation.edge_mask.detach().cpu().numpy()
    
    return node_mask, edge_mask

def plot_explanation_plotly(data, raw_coords, node_mask, edge_mask):
    """
    Visualizes the graph explanation using Plotly.
    Nodes and edges are colored/sized based on their importance mask.
    """
    if node_mask is None:
        return go.Figure()
        
    edge_index = data.edge_index.cpu().numpy()
    
    # Create figure
    fig = go.Figure()
    
    # 1. Plot Edges
    # To optimize Plotly performance, we group edges or just plot them individually if small
    # For explanation, we can map edge_mask to opacity or color
    # Normalize edge mask
    if len(edge_mask) > 0 and edge_mask.max() > 0:
        edge_mask_norm = edge_mask / edge_mask.max()
    else:
        edge_mask_norm = edge_mask
        
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        x0, y0 = raw_coords[src]
        x1, y1 = raw_coords[dst]
        
        weight = edge_mask_norm[i]
        color = f"rgba(255, 107, 107, {max(0.1, weight)})" # Red with varying opacity
        
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=1 + (weight * 3), color=color),
            hoverinfo='none',
            showlegend=False
        ))
        
    # 2. Plot Nodes
    if len(node_mask) > 0 and node_mask.max() > 0:
        node_mask_norm = node_mask / node_mask.max()
    else:
        node_mask_norm = node_mask
        
    sizes = 8 + (node_mask_norm * 20) # Sizes between 8 and 28
    
    fig.add_trace(go.Scatter(
        x=raw_coords[:, 0],
        y=raw_coords[:, 1],
        mode='markers',
        marker=dict(
            size=sizes,
            color=node_mask_norm,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance"),
            line=dict(color='white', width=1)
        ),
        text=[f"Node {i}<br>Importance: {val:.3f}" for i, val in enumerate(node_mask)],
        hoverinfo='text',
        name="Fixations"
    ))
    
    fig.update_layout(
        title="GNNExplainer: Fixation Importance",
        plot_bgcolor='rgba(26,31,46,0.4)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8892A0'),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, autorange='reversed'), # Reverse Y for screen coords
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig
