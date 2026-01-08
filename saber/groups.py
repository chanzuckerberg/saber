import rich_click as click

# Configure rich-click
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True

# Define commonly used SAM2-AMG flags for reuse
amg_flags = [
	"--sam2-cfg", 
	"--npoints", "--points-per-batch",
	"--pred-iou-thresh", "--box-nms-thresh",
	"--crop-n-layers", "--crop-n-points",
	"--use-m2m", "--multimask",
]

# 
click.rich_click.COMMAND_GROUPS = {
	"routines classifier": [
		{"name": "Pre-Process", 
		"commands": ["prep2d", "prep3d", "split-data", "merge-data", "labeler"]
		},
		{"name": "Train Classifier", "commands":["train"]},
		{"name": "Inference", "commands": ["predict", "evaluate"]}
	]
}

click.rich_click.OPTION_GROUPS = {
    "routines classifier prep2d": [
        {"name": "I/O", "options": ["--input", "--output", "--scale-factor", "--target-resolution"]},
        {"name": "SAM2-AMG", "options": [ *amg_flags ]},
    ],
	"routines classifier prep3d": [
		{"name": "I/O", "options": ["--config", "--voxel-size", "--tomo-alg", "--output"]},
		{"name": "Initialize Slabs", "options": ["--num-slabs", "--slab-thickness"]},
		{"name": "SAM2-AMG", "options": [ *amg_flags ]},
	],
	"routines segment slab": [
		{"name": "Input", "options": ["--config", "--voxel-size", "--tomo-alg", "--run-id", "--slab-thickness"]},
	  	{"name": "Classifier", "options": ["--model-weights", "--model-config", "--target-class"]},
		{"name": "SAM2-AMG", "options": [ *amg_flags ]}, 
	],
	"routines segment micrographs": [
		{"name": "I/O", "options": ["--input", "--output", "--scale-factor", "--target-resolution", "--sliding-window"]},
	  	{"name": "Classifier", "options": ["--model-weights", "--model-config", "--target-class"]},
		{"name": "SAM2-AMG", "options": [ *amg_flags ]},
	],
	 "routines segment tomograms": [
		{"name": "Input", "options": ["--config", "--voxel-size", "--tomo-alg", "--run-ids", "--slab-thickness", "--multi-slab"]},
	  	{"name": "Classifier", "options": ["--model-weights", "--model-config", "--target-class"]},
		{"name": "Output", "options": ["--seg-name", "--seg-session-id" ]},
	 ],
	 "routines segment fib": [
		{"name": "I/O", "options": ["--input", "--scale-factor", "--output"]},
		{"name": "Processing Parameters", "options": ["--ini_depth", "--nframes",]},
		{"name": "Classifier", "options": ["--model-config", "--model-weights", "--target-class"]}
	 ]
,
	 "routines segment light": [
		{"name": "I/O", "options": ["--input", "--scale-factor", "--output"]},
		{"name": "Processing Parameters", "options": ["--ini_depth", "--nframes",]},
		{"name": "Classifier", "options": ["--model-config", "--model-weights", "--target-class"]}
	 ]	 
}