Dot brightness distribution rule
When sampling a dot using a circular or rectangular shape, the area that does not belong to the dot is considered the Background of Dot (BOD).
Calculate the average background brightness B from the BOD region.
For each dot pixel with brightness L, define its darkness as:
D = B - L
From multiple samples, learn the distribution:
P(D | B, r)
where r is the relative position of the pixel from the dot center to its edge.
When generating a new dot, use the BOD brightness and position r to obtain an appropriate darkness value, then:
L = B - D
Rule for overlapping dots
When two dots affect the same pixel:
Da = BOD - La
Db = BOD - Lb
The darkness in the overlapping region is the sum of the darkness values:
Dab = Da + Db
The final brightness is:
Lab = BOD - (Da + Db)
equivalently:
Lab = La + Lb - BOD
Lab must not be darker than the minimum allowed brightness of the two dots:
Lab = max(Lfloor, La + Lb - BOD)
where Lfloor is the darkest pixel value of the two dots.
Value sliders
The value sliders in Tab 1 must be fixed.
The value sliders in Tab 4 must allow adjustment of Min, Mean, and Max values. Currently, these values cannot be adjusted.