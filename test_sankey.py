import plotly.graph_objects as go
import numpy as np

labels = ["Baseline 1", "Baseline 2", "Baseline 3", "Baseline 4", "Baseline 5"]
x = [0.0]*5
y = [0.1, 0.3, 0.5, 0.7, 0.9]

for i in range(1, 6):
    labels.extend([f"Male {i}", f"Female {i}", f"NB {i}"])
    x.extend([1.0]*3)
    # small offsets
    base_y = -0.1 + i * 0.2
    y.extend([base_y, base_y + 0.05, base_y + 0.1])

fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        label=labels,
        x=x,
        y=y,
        pad=10
    ),
    link=dict(
        source=[0, 1, 2],
        target=[5, 9, 13],
        value=[10, 20, 30]
    )
)])
fig.write_image("test_sankey.png")
