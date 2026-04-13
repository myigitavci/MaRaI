# Metadata Text Format (Recommended)


## Recommended template

Use this pattern:

`A brain MRI, plane <plane>, Scanner (Manufacturer, Model, Field Strength): (<manufacturer>, <model>, <field_strength>), Acquisition (Description, Sequence, Variant): (<description>, <sequence>, <variant>), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (<TE>, <TR>, <TI or NONE>, <FA>)`

## Example (T2w)

`A brain MRI, plane axial, Scanner (Manufacturer, Model, Field Strength): (GE, SIGNA_HDx, 1.5), Acquisition (Description, Sequence, Variant): (Ax T2, SE, SK), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (0.10192, 5.36, NONE, 90)`

## Notes

- Keep units and conventions consistent within your dataset.
- If a field is unavailable, use `NONE`.
- Prefer real DICOM-derived metadata over approximations.
