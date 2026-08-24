RANDOM DIRECTIONAL INK BLUR / SMEAR AUGMENTATION FRAMEWORK

Goal:
Create realistic printing defects where ink pixels are blurred, smeared, spread, or distorted in locally varying directions, while keeping the background unaffected. The augmentation should operate on the rendered ink mask or character layer before compositing it onto the final background.

1. PATCH-WISE RANDOM MOTION BLUR

Method:
- Divide the ink region into small overlapping patches.
- Assign each patch a random blur angle and blur length.
- Apply a directional motion-blur kernel independently to each patch.
- Blend overlapping patches smoothly to avoid visible block boundaries.

Main parameters:
- patch_size
- patch_overlap
- blur_angle_range
- blur_length_range
- blur_strength

Advantages:
- Simple to implement.
- Fast.
- Easy to control.
- Suitable for local ink smearing.

Possible issue:
- Hard patch boundaries may appear if patches are not overlapped or blended smoothly.


2. SPATIALLY VARYING DIRECTIONAL BLUR / VECTOR-FIELD BLUR

Method:
Create two spatially varying maps:

    angle_map(x, y)
    length_map(x, y)

where:

- angle_map defines the local blur direction.
- length_map defines the local blur distance.

Each image location is blurred according to its own local direction and length.

Conceptually:

    I'(x, y) =
        sum I(x-u, y-v) *
        K(angle_map(x,y), length_map(x,y))

Important:
The maps should NOT contain completely independent random values for every pixel.

Instead, generate smooth spatial fields using methods such as:
- Gaussian-smoothed random noise
- low-frequency noise
- Perlin noise
- interpolated random control points

This makes nearby pixels have similar directions and produces physically plausible printing defects.

Advantages:
- Most flexible method.
- Produces natural local variation.
- Can simulate irregular ink drag, roller effects, and localized printing defects.


3. ANISOTROPIC GAUSSIAN BLUR

Method:
Use an elliptical Gaussian kernel instead of a circular Gaussian kernel.

The kernel has:
- major-axis sigma
- minor-axis sigma
- orientation angle

Example:

    sigma_long > sigma_short

The long axis defines the dominant blur direction.

For spatial variation, allow:
- orientation to change across the image
- sigma_long to change across the image
- sigma_short to change across the image

Main parameters:
- sigma_long
- sigma_short
- angle
- local_strength

Advantages:
- Produces smooth directional blur.
- Less harsh than motion blur.
- Suitable for soft ink spreading and mild directional smearing.

Limitation:
- Usually looks less like strong mechanical ink dragging than motion blur.


4. RANDOM LINE-SPREAD / STROKE-SMEAR MODEL

Method:
Treat each ink pixel or local ink sample as a small amount of ink that can spread along a short directional path.

For each ink location:
- determine a local direction
- determine a smear distance
- redistribute part of its intensity along that direction

The local smear can be represented as:

    P(t) = ink contribution along direction theta

for:

    0 <= t <= smear_length

Optionally use a decay function:

    P(t) = exp(-t / lambda)

so the smear becomes weaker as it moves away from the source pixel.

Main parameters:
- smear_angle
- smear_length
- smear_decay
- smear_strength

Advantages:
- Very suitable for ink-specific simulation.
- Can produce asymmetric tails.
- Better than standard blur when the defect looks like dragged or transferred ink.


5. ANISOTROPIC DIFFUSION

Method:
Simulate directional ink diffusion using a diffusion tensor.

Instead of allowing ink to spread equally in every direction, define:

    D_parallel
    D_perpendicular

where:

    D_parallel > D_perpendicular

The diffusion direction may vary spatially according to an orientation field.

Advantages:
- Produces physically smooth directional spreading.
- Can simulate progressive ink migration.

Limitations:
- More computationally expensive.
- More difficult to tune.
- Usually unnecessary if motion blur or vector-field smear already produces the required appearance.


6. SMOOTH RANDOM ORIENTATION FIELD

This should be used by the more advanced methods above.

Do NOT generate:

    theta(x,y) = independent random angle per pixel

because it creates noisy, unrealistic deformation.

Instead:

1. Generate low-resolution random noise.
2. Upsample it smoothly.
3. Apply Gaussian smoothing.
4. Convert the resulting field into an angle range.

Example:

    theta(x,y) =
        theta_mean +
        theta_variation * smooth_noise(x,y)

Similarly:

    L(x,y) =
        L_min +
        smooth_noise(x,y) * (L_max - L_min)

This creates coherent defect regions where nearby dots are affected similarly.


7. INK BLEED / MORPHOLOGICAL SPREAD

Directional blur can be combined with ink expansion.

Possible methods:
- dilation
- directional dilation
- distance-transform-based bleeding
- probabilistic outward ink spread

A more realistic bleed model can use distance from the existing ink boundary:

    P_ink(d) =
        exp(-d² / (2 * sigma²))

where:
- d = distance from the original ink mask
- sigma = bleed radius

This allows irregular ink expansion instead of perfectly uniform morphological dilation.


8. DOT DROPOUT AND PARTIAL DOT DAMAGE

For dot-matrix printing, apply defects at the dot level before directional blur.

Possible operations:
- remove complete dots
- reduce dot opacity
- erode part of a dot
- randomly remove a section of a dot
- deform the dot mask
- vary dot darkness

Parameters:
- dot_dropout_probability
- dot_alpha_range
- dot_erosion_range
- partial_damage_probability
- partial_damage_strength


9. SPATIAL INK-STRENGTH FIELD

Printing defects often affect neighboring dots together.

Create a smooth field:

    A(x,y)

and modify the local ink darkness:

    Ink_new(x,y) =
        Ink_original(x,y) * A(x,y)

The field should vary slowly across the printed region.

This can create:
- local faded regions
- local over-inking
- bands of weak printing
- clusters of damaged dots

Recommended generation methods:
- Gaussian-smoothed random noise
- low-frequency random field
- Perlin noise


10. RECOMMENDED COMBINED PIPELINE

Recommended order:

    PCA-generated dot
        ↓
    per-dot darkness variation
        ↓
    dot dropout / partial dot damage
        ↓
    assemble full character or text ink mask
        ↓
    spatial ink-strength field
        ↓
    ink bleed / directional dilation
        ↓
    vector-field motion smear
        ↓
    anisotropic blur
        ↓
    local final blur
        ↓
    composite onto background

Important:
Apply these transformations to the INK LAYER only.

Do NOT apply them to the complete final image unless the real physical defect also affects the background.


11. RECOMMENDED PARAMETERS

Typical starting ranges for small dot-matrix characters:

    blur_angle_range       = -20° to +20°
    blur_length_range      = 1 to 6 px
    anisotropic_sigma_long = 0.5 to 3.0 px
    anisotropic_sigma_short= 0.1 to 1.0 px
    bleed_radius           = 0 to 3 px
    smear_strength         = 0.2 to 0.8
    dot_dropout_probability= 0.00 to 0.15
    dot_alpha              = 0.3 to 1.0
    local_field_scale      = large enough to affect groups of nearby dots

These ranges should ultimately be fitted from real defective samples instead of being chosen arbitrarily.


12. RECOMMENDED IMPLEMENTATION PRIORITY

Level 1 — Simple:
    Patch-wise random motion blur

Level 2 — Better:
    Smooth vector-field motion blur

Level 3 — More realistic:
    Smooth vector field
    + ink bleed
    + spatial ink-strength variation
    + per-dot damage

Level 4 — Physical-style simulation:
    Directional line-spread
    + anisotropic diffusion
    + learned parameter distributions from real NG images


13. CORE PRINCIPLE

The most realistic result is not produced by making every pixel independently random.

Instead, use spatially correlated randomness:

    nearby pixels → similar direction
    nearby dots   → similar defect strength
    different regions → gradually different behavior

This creates coherent printing defects that resemble real mechanical, ink-transfer, pressure, and surface-contact problems rather than synthetic image noise.