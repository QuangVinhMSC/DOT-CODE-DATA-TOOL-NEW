Add a new tab called Defect Generation between "create job" and "class definition" tab.

This tab will allow the user to create different types of defects that can affect individual characters or entire text lines.

Examples of supported defects include:

- Remove the upper or lower portion of a text line. like @toplost.png and @botlost.png
- Create ink smearing that spreads over and overlaps multiple characters. like @coverink.png
- Remove several characters from a line. like @randomlost.png
- Compress or collapse an entire line toward one side until it becomes a concentrated ink blob. like @dfall.png
- Collapse only one side of a line toward a specific direction. like @df1side.md
- Horizontally compress an entire line so that all characters become narrower. like @dfscalde.md

Any character that is directly affected by a defect must not receive a bounding box.
For every defect type that is enabled, create an additional line-level class representing that defect.