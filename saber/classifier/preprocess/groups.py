import rich_click as click

# Configure rich-click
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True

click.rich_click.OPTION_GROUPS = {
    "routines classifier prep2d": [
        {"name": "I/O", "options": ["--input", "--output", "--scale-factor", "--target-resolution"]},
        {"name": "SAM2-AMG", "options": [
            "--sam2-cfg",
            "--npoints",
            "--points-per-batch",
            "--pred-iou-thresh",
            "--crop-n-layers",
            "--box-nms-thresh",
            "--crop-n-points",
            "--use-m2m",
            "--multimask",
        ]},
    ],
	"routines classifier prep3d": [
		{"name": "I/O", "options": ["--config", "--voxel-size", "--tomo-alg", "--output"]},
		{"name": "Initialize Slabs", "options": ["--num-slabs", "--slab-thickness"]},
		{"name": "SAM2-AMG", "options": [
			"--sam2-cfg",
			"--npoints",
			"--points-per-batch",
			"--pred-iou-thresh",
			"--crop-n-layers",
			"--box-nms-thresh",
			"--crop-n-points",
			"--use-m2m",
			"--multimask",
		]},
	],
}