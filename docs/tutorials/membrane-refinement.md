# Membrane Refinement

Membrane refinement is a crucial post-processing step that improves the quality and consistency of organelle and membrane segmentations. This tutorial covers how to use SABER's GPU-optimized membrane refinement pipeline to clean up and enhance your segmentation results.

---

## 🔬 Understanding Membrane Refinement

### Why Refine Membranes?

Raw segmentation outputs often contain artifacts and inconsistencies that can affect downstream analysis:

- **Boundary artifacts**: Segmentation errors at image edges
- **Small noise objects**: Tiny false positive detections
- **Topological inconsistencies**: Organelles extending beyond membrane boundaries
- **Surface membrane confusion**: Internal membrane fragments that should be removed

### The Combined Mask Approach

SABER's membrane refinement uses an innovative "combined mask" strategy:

<details markdown="1">
<summary><strong>How the combined mask approach works</strong></summary>

1. **Membrane subtraction**: Membrane pixels are subtracted from organelle pixels
2. **Interior creation**: This creates a clean interior mask for each organelle
3. **Morphological cleaning**: Opening operations remove small artifacts
4. **Constraint application**: The cleaned interior constrains both organelle and membrane
5. **Topological consistency**: Ensures organelles stay within membrane boundaries

This approach ensures that:
- Organelles are properly contained within their membranes
- Membranes are refined to match their corresponding organelles
- Both segmentations maintain topological consistency
- Small artifacts and noise are effectively removed

</details>

---

## 🚀 Running Membrane Refinement

### Basic Command

Refine organelle and membrane segmentations using the CLI:

```bash
saber analysis refine-membranes \
    --config config.json \
    --org-info "organelles,saber,1" \
    --mem-info "membranes,membrane-seg,1" \
    --voxel-size 10 \
    --save-session-id "refined-1"
```

### Parameter Explanation

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--config` | Path to copick config file | `config.json` | `my_project.json` |
| `--org-info` | Organelle segmentation info | `organelles,saber,1` | `mitochondria,user1,session2` |
| `--mem-info` | Membrane segmentation info | `membranes,membrane-seg,1` | `mito_membranes,user1,session2` |
| `--voxel-size` | Voxel size in Angstroms | `10` | `5.2` |
| `--save-session-id` | Session ID for refined results | `1` | `refined-v2` |

### Information Format

The `--org-info` and `--mem-info` parameters use a specific format:

```bash
# Format: "name,userID,sessionID"
--org-info "mitochondria,user1,session2"

# If no userID/sessionID specified, uses defaults
--org-info "organelles"  # Uses default user and session
```

---

## 🔧 Customizing Refinement Parameters

### Using the Analysis Module Directly

For custom refinement workflows, you can use the analysis module directly:

```python
from saber.analysis.refine_membranes import OrganelleMembraneFilter, FilteringConfig

# Create custom configuration
config = FilteringConfig(
    ball_size=5,              # Morphological operation kernel size
    min_membrane_area=10000,  # Minimum membrane component size
    edge_trim_z=5,           # Z-edge trimming pixels
    edge_trim_xy=3,          # XY-edge trimming pixels
    batch_size=8,            # GPU batch processing size
    keep_surface_membranes=False  # Remove internal membranes
)

# Initialize filter
filter_obj = OrganelleMembraneFilter(config, gpu_id=0)

# Run refinement
results = filter_obj.run(organelle_seg, membrane_seg)

# Access results
refined_organelles = results['organelles']
refined_membranes = results['membranes']
```

### Configuration Parameters

<details markdown="1">
<summary><strong>Detailed parameter descriptions</strong></summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ball_size` | int | 3 | Radius of morphological operation kernel |
| `min_membrane_area` | int | 10000 | Minimum area for membrane components |
| `edge_trim_z` | int | 5 | Pixels to trim from Z edges |
| `edge_trim_xy` | int | 3 | Pixels to trim from XY edges |
| `min_roi_relative_size` | float | 0.15 | Minimum ROI size relative to organelle |
| `batch_size` | int | 8 | GPU batch processing size |
| `keep_surface_membranes` | bool | False | Keep only surface membranes |

**Parameter tuning guidelines**:

- **`ball_size`**: Larger values create smoother boundaries but may remove fine details
- **`min_membrane_area`**: Adjust based on your expected membrane component sizes
- **`edge_trim_*`**: Increase if you have boundary artifacts
- **`keep_surface_membranes`**: Enable to remove internal membrane fragments

</details>

---

## 📊 Understanding the Results

### Output Structure

The refinement process produces:

1. **Refined organelle segmentation**: Cleaned and constrained organelle labels
2. **Refined membrane segmentation**: Surface membranes only (if enabled)
3. **Topologically consistent results**: Organelles properly contained within membranes

### Quality Assessment

<details markdown="1">
<summary><strong>How to assess refinement quality</strong></summary>

**Good refinement signs**:

- Clean, well-defined organelle boundaries
- Membranes properly surrounding organelles
- Removal of small noise objects
- Consistent topology between organelles and membranes
- No boundary artifacts

**Potential issues to watch for**:

- Over-aggressive cleaning removing valid structures
- Incomplete membrane coverage around organelles
- Loss of fine structural details
- Inconsistent results across different regions

**Troubleshooting**:

- **Too much cleaning**: Reduce `ball_size` or `min_membrane_area`
- **Not enough cleaning**: Increase parameters or enable `keep_surface_membranes`
- **Boundary artifacts**: Increase `edge_trim_*` parameters
- **GPU memory issues**: Reduce `batch_size`

</details>

---

## 🎯 Best Practices

### When to Use Membrane Refinement

**Use refinement when**:
- You have both organelle and membrane segmentations
- Raw segmentations contain noise or artifacts
- You need topologically consistent results
- Downstream analysis requires clean boundaries

**Consider skipping refinement when**:
- Segmentations are already very clean
- You only have organelle OR membrane data (not both)
- You need to preserve all fine details
- Processing time is critical

### Parameter Selection Guidelines

1. **Start with defaults**: The default parameters work well for most cases
2. **Adjust based on data**: Modify parameters based on your specific data characteristics
3. **Iterate gradually**: Make small parameter changes and assess results
4. **Consider your goals**: Balance between cleaning and detail preservation

---

## 🚀 What's Next?

After membrane refinement, you can:

- **Analyze refined results**: Use the cleaned segmentations for quantitative analysis
- **Visualize results**: Create galleries and visualizations of refined segmentations
- **Export for other tools**: Use refined segmentations in external analysis pipelines
- **Iterate and improve**: Adjust parameters based on results and re-run if needed

**Integration with other SABER tools**: Refined segmentations work seamlessly with SABER's analysis and visualization modules.


