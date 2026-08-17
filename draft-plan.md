TAB 1 — SAMPLE COLLECTION
1. Sample Image Management
Multiple images can be uploaded as samples.
Maximum number of sample images: 5 images.
The images are displayed in the same frame.
Use mini tabs to switch between images.
Sample images can be zoomed up to 500 times.
Samples taken from different images are shared together.
2. Dot Sample Collection Tools

There are 3 drawing tools:

Circle.
Rectangle.
Closed outline using a solid line.

The three tools are located at the top-left corner of the sample image, arranged vertically downward.

All three tools can be used for sample collection.

The content inside the outline is added to the “dot samples” section.

Maximum: 10 dot samples.
Fewer than 10 samples can be used to complete dot reconstruction.

The dot sample processing and dot reconstruction have been tested in:

test1.py

3. Dot Sample Set Parameters

Next to the sample image area is the parameter bar of the sample set after calculation is completed.

The parameters include the value ranges required for dot reconstruction as in:

test1.py

The parameter bars use the following display:

1 red dot in the middle: Mean.
2 blue dots on both sides: Min and Max.
4. Reconstructed Dot Test Area

Directly below the parameter section is a small test panel.

The test panel:

Can use a white background by default.
Or the user can upload a background.

The user can click directly on the test panel to see how the dot is reconstructed.

5. Rectangle Tool for Calculating Perspective Rules and Tilt

In addition to sample collection, the rectangle drawing tool also has another function.

The user can:

Adjust each corner point of the rectangle.
Use the corner points to calculate the perspective rule.
At the same time, use them as the basis for determining:
X tilt;
Y tilt.

The parameters of the perspective rule are displayed in the parameter panel.

These parameters:

Can be used.
Or can be unused.

This part must be implemented separately so that it can be tested directly and the calculation method can be validated.

6. Curve Tool

There is an additional curve drawing tool.

Only 2 curves are required to calculate the waviness of the number line.

This is suitable for numbers located on a cylindrical surface.

This parameter:

Can be used.
Or can be unused.
7. Background Separation Tool for Background Sampling

There is an additional background separation tool for background sampling.

Initially, two parameters are collected:

Brightness.
Contrast.

The tool uses a Threshold bar so the user can manually separate the characters.

The removed pixels must be filled in using the surrounding background.

This part does not need to be implemented immediately.

8. Tool for Measuring the Distance Between Dots

There is an additional tool to measure the distance between dots.

The user samples pairs of dots to measure:

Horizontal distance.
Vertical distance.

The results are transferred to Tab 2 as two parameters:

Horizontal distance.
Vertical distance.

The parameters in Tab 2 are value ranges calculated from multiple pairs of dots.

TAB 2 — NUMBER MATRIX

This tab is similar to Tab 1 of:

D:\I&C\DOT-CODE-DATA-TOOL

1. Character Matrix Configuration

In this tab, the user can adjust the configuration of each character as before.

In addition, the following can also be used:

Vertical distance obtained from Tab 1.
Horizontal distance obtained from Tab 1.
2. Defining the Distance Between Dots

For example, with the number 1:

If there are 2 dots separated by 1 cell vertically, the user can define the distance between these two dots as:

1 × “vertical distance”

After this has been defined, pairs of dots in the number 1 that are 1 cell apart vertically will all have a distance of:

1 × “vertical distance”

When switching to the definition of another number, the user can still define it differently.

The numbers are not required to use the same definition.

3. Mouse Operations on the Matrix
Left Mouse Button

Use the left mouse button to place dots on the matrix.

Right Mouse Button

Use the right mouse button to select placed dots.

After a pair of dots has been selected, the user can directly enter the coefficient multiplied by the vertical/horizontal distance unit.

Vertical relation → multiply by vertical distance.
Horizontal relation → multiply by horizontal distance.

Press:

Enter

to finish entering the coefficient.

4. Displaying the Coefficient Directly on the Matrix

The coefficient is displayed directly on the connecting line between the two dots.

Example:

.<----2----->.

5. Reselecting Dots

If 2 dots have already been selected but the user wants to select again:

Esc

6. Deleting a Coefficient Link

To delete a coefficient link between two dots:

Click directly on the connecting line.
Press:

Delete

7. Mandatory Constraint for Each Number

A number must have:

1 vertical constraint.
1 horizontal constraint.

It cannot have:

Fewer.
More.
8. Saving the Number Format

Below the matrix there is a button:

“Save”

Used to save the format of that number into the current process.

9. Displaying Value Ranges

The parameter bars in Tab 2 use:

Red dot: Mean.
Two blue dots: Min and Max.
TAB 3 — SUMMARY
1. Character Display Area

This tab has 2 image frames, positioned one above the other.

Each frame occupies:

50% of the vertical space.

Both frames display a character selected by the user.

The two frames:

Display the same character.
But differ based on the parameters selected on the right side.
2. Parameter Summary Area

The remaining space on the right summarizes all parameter bars from Tab 1 and Tab 2.

Each bar continues to display:

Mean.
Min.
Max.

According to the convention:

Red dot = Mean.
Two blue dots = Min and Max.
3. Selecting Min/Max Parameters for Comparison

Next to each parameter bar is a checkbox.

If a bar is selected:

The upper image uses the Min value of the bar.
The lower image uses the Max value of the bar.

If the bar is not selected:

Both the upper and lower images use the Mean value.

Multiple bars can be selected at the same time.

4. Saving the Configuration

At the bottom of the tab there is a button:

“Save configuration”

The user can save the configured settings into a file so that the program can load them again the next time it is used.

TAB 4 — CREATE JOB
1. Background Management

The user can upload:

1 background.
Or multiple backgrounds.

The size of the image is displayed in a small box directly below the image.

2. Resize Background

The user can change the size using this box.

After resizing and pressing:

“Save size”

the entire background set must be resized to the same size.

If the aspect ratios of the images are different:

They must be cropped automatically.
The images must not be stretched.
Cropping takes the center part.
An image can only be cropped on 1 dimension.
3. Base Quadrilateral

On each background, the user is allowed to draw 1 quadrilateral using the quadrilateral drawing tool as in Tab 1.

Temporarily call this quadrilateral:

“Base quadrilateral”

All backgrounds must contain 1 base quadrilateral before proceeding to the next step.

4. Line and Character Management

Next to the background display bar, there is an area for managing the creation of:

Characters.
Lines.

The characters and lines are displayed directly at the center of the background.

At this step, the specific position does not need to be considered yet.

5. Line Spacing

Each pair of lines must have a connecting line in the form:

<----2----->

The user can directly adjust the line spacing through this connecting line.

6. Character Spacing

Character spacing is entered through a box located outside the background frame.

7. Replacement Characters

Each entered character must have its own management bar.

This bar allows the user to select additional characters that are allowed to appear as replacements for that character.

The user can select:

One character.
Or multiple characters.
8. Defective Dots

Below are 3 items to define, temporarily referred to collectively as:

“Defective dots”

Type 1 — Missing Dots

Define:

The maximum number of dots that a character can lose.
The probability of the event of 1 dot being missing per character.
Type 2 — Deformed Dots

Define:

The maximum number of deformed dots in a character.
The probability of the event of 1 dot being deformed per character.
Type 3 — Strongly Jittered Dots

Define:

The maximum number of strongly jittered dots in a character.
The strong jitter level around the position.
The probability of the event of 1 dot strongly jittering.

If a type of defective dot is not used, set the value of that type to:

0

Defective dots do not need to be displayed immediately.

Temporarily call the maximum numbers of defective dots of the three types respectively:

m, n, l

TAB 5 — CLASS DEFINITION
1. Load Class

There is a button:

“Load class”

This button is used to load information from Tab 4.

2. Character Classes

List classes for all characters.

Temporarily call the number of character classes:

x classes

There are an additional:

x classes

representing failed characters.

Thus, each character can have:

Pass class.
Fail class.
3. Failed Character Classes

Failed character classes can be:

Kept.
Or disabled so that they disappear from the class list.

If the failed class is still kept, the following must be defined:

The minimum number of defective dots required for the character to become a failed character.

Note: in the exceptional case where a character keeps only 1 class and that class is the failed-character class, no condition is required.

4. Line Classes

Line classes are listed according to the number of lines.

Example:

line1

line2

The line classes can be directly classified as:

Pass.
Or fail.

There is no need to create an additional opposite class.

For example, if there are 2 lines, there are only:

line1

line2

meaning there are only 2 line classes.

5. Delete Class

The classes are listed after pressing:

“Load class”

But the user can still delete classes from the list.

Note: the characters selected as replacement characters in Tab 4 also need to have classes.

TAB 6 — SAVE JOB AND EXPORT DATA
1. Data Format

This tab has standard data formats such as YOLO for the user to choose from.

2. Save Job

There is a button:

“Save job”

Used to save the current data configuration defined from:

Tab 1.
Tab 2.
Tab 3.
Tab 4.
Tab 5.

After saving the job, the user can return to Tab 1 to define a new job.

3. Export Data

When enough required jobs have been created, the user presses:

“Export data”

to export the entire dataset.

GENERAL RULES
1. Dataset Classes

The classes of an exported dataset are:

All classes that appear in all jobs.

2. Initial Positions of Characters and Lines

When initialized, characters and lines must:

Be located inside the base quadrilateral.
Have their specific positions randomized.
3. Parameter Randomization

The parameter adjustment bars mentioned in Tab 1 and Tab 2 all have ranges:

Min → Max

When exporting data, the actual values are:

Randomized within the Min → Max range.

4. Bounding Box

All:

Characters.
Lines.

must have appropriate bounding boxes, ensuring that YOLO data can be exported smoothly.