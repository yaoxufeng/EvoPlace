# Evolution of Optimization Algorithms for Global Placement via Large Language Models

Built upon the GPU-accelerated global placer [DREAMPlace](https://doi.org/10.1109/TCAD.2020.3003843)

# Dependency 

- [DREAMPlace](https://github.com/limbo018/DREAMPlace)
    - Commit 105ff790474e91058da22281624d87e8a1922196 (default 4.1 version)

# Installed dreamplace directory structure
All experiments are conducted in the installed DREAMPlace directory (`dreamplace_install`). We provide our `dreamplace_install` directory in this anonymous GitHub repository.
<pre>
project-root/
├── <strong>build/</strong>               
├── <strong>cmake/</strong>             
├── <strong>dreamplace/</strong>            
├── <strong>dreamplace_install</strong>    # installed dreamplace directory          
│   ├── benchmarks            # downloaded benchmarks 
│   ├── dreamplace            # dreamplace directory  
│   └── total_best_configs    # discovered algorithms
│       ├── mms               # mms discovered algorithms
│       └── ispd2005free      # ispd2005 discovered algorithms
│       └── ispd2019          # ispd2019 discovered algorithms
│   └── include   
│   └── bin    
│   └── ...
├── ...                 
└── README.md               # Project readme
</pre>


The main algorithms are implemented in:
- `dreamplace/BasicPlace.py` - Initialization algorithms
- `dreamplace/PlaceObj.py` - Preconditioner algorithms
- `dreamplace/CustomOptimizer.py` - Optimizer algorithms (includes default Nesterov implementation)

Backup files are provided to maintain baseline results:
```bash
cp ./dreamplace/BasicPlace_backup.py ./dreamplace/BasicPlace.py
cp ./dreamplace/PlaceObj_backup.py ./dreamplace/PlaceObj.py
cp ./dreamplace/CustomOptimizer_backup.py ./dreamplace/CustomOptimizer.py
```

The discovered algorithms are all py file and the results can also be achieved by replace algorithm py file. 
The discovered algorithms are in:

<pre>
├── <strong>dreamplace_install</strong>    # installed dreamplace directory          
│   ├── dreamplace                      # dreamplace directory 
│       ├── BasicPlace.py               # containing init algorithm
│       └── BasicPlace_backup.py        
│       └── PlaceObj.py                 # containing prec algorithm
│       └── PlaceObj_backup.py      
│       └── CustomOptimizer.py          # containing opt algorithm
│       └── CustomOptimizer_backup.py         
│       └── ...      
│   └── total_best_configs    # discovered algorithms
│       ├── mms               # mms discovered algorithms
│           ├── adaptec1               # case adaptec1 
│               ├── best_MacroInit.py  # discovered init algorithms
│           └── adaptec2               # case adaptec2
│           ├── adaptec3               # case adaptec3 
│           └── ...  
│       └── ispd2005free      # ispd2005 discovered algorithms
│       └── ispd2019          # ispd2019 discovered algorithms
</pre>

You can simply replace original algorithm file and obtain associated case's results:
```
cp total_best_configs/mms/adaptec1/best_MacroInit.py dreamplace/BasicPlace.py
python dreamplace/Placer.py test/mms_llm4placement/adaptec1.json
```

Note that the only difference between our configs directory mms_llm4placement and original one mms is we replace nesterov optimizer args with custom setting.

All prompts and evolution files are included in:
<pre>
├── <strong>dreamplace_install</strong>    # installed dreamplace directory          
│   ├── prompt                             # all prompts
│       ├── macro_init                 
│       └── macro_init_improve        
│       └── precondition                 
│       └── ...     
├── llm_macro_init.py                 
└── llm_macro_evolution.py   
├── llm_precondition_init.py                 
└── llm_precondition_evolution.py
├── llm_optimizer_init.py               
└── llm_optimizer_evolution.py
└── ...            
</pre>

Following is an example of prompt template
<pre>
You are a distinguished professor specializing in optimization, and electronic design automation.

# Task: Improve Dreamplace Macro Init

## Context:
- Problem: Analytical global placement in electronic design automation
- Tool: DreamPlace
- Objective: Improve Half-Perimeter Wirelength (HPWL) and Fast Convergence

## Background
In the context of VLSI global placement, the goal is to determine the optimal positions of standard cells, macros, and pins on a chip to minimize objectives such as wirelength and overlap, while respecting constraints such as density.
Global placement: Cells are placed in a continuous domain, focusing on minimizing wirelength and spreading cells to avoid overlaps.
Detailed placement: Adjustments are made in a discrete domain to finalize the placement.
Key Terms:
Cells: These are the functional units (such as logic gates or flip-flops) that need to be placed on the chip.
Pins: Each cell has pins that represent connections to other cells.
Nets: A net is a collection of pins that need to be connected, forming the routing between cells.
Wirelength: This is the total length of the nets, and minimizing it is a primary objective to reduce delay and power consumption.


## Dreamplace Macro Init:
```python
class BasicPlace(nn.Module):
    def __init__(self, params, placedb, timer):
        torch.manual_seed(params.random_seed)
        super(BasicPlace, self).__init__()
        tt = time.time()
        self.init_pos = np.zeros(placedb.num_nodes * 2, dtype=placedb.dtype)
        # x position
        self.init_pos[0:placedb.num_physical_nodes] = placedb.node_x
        if params.global_place_flag and params.random_center_init_flag:  # move to center of layout
            logging.info(
                "move cells to the center of layout with random noise")
            self.init_pos[0:placedb.num_movable_nodes] = np.random.normal(
                loc=(placedb.xl * 1.0 + placedb.xh * 1.0) / 2,
                scale=(placedb.xh - placedb.xl) * 0.001,
                size=placedb.num_movable_nodes)

        # y position
        self.init_pos[placedb.num_nodes:placedb.num_nodes +
                      placedb.num_physical_nodes] = placedb.node_y
        if params.global_place_flag and params.random_center_init_flag:  # move to center of layout
            self.init_pos[placedb.num_nodes:placedb.num_nodes +
                          placedb.num_movable_nodes] = np.random.normal(
                              loc=(placedb.yl * 1.0 + placedb.yh * 1.0) / 2,
                              scale=(placedb.yh - placedb.yl) * 0.001,
                              size=placedb.num_movable_nodes)
```


## Analysis of Current Dreamplace Macro Initialization
@@Macro-Init-Ana@@

## Available placedb objects:
placedb.node_size_x:  # 1D array, cell width, length: 371462
placedb.node_size_y:  # 1D array, cell height, length:  371462
placedb.pin_offset_x: # 1D array, pin offset x to its node, length:  919701
placedb.pin_offset_y: # 1D array, pin offset y to its node, length:  919701
placedb.net_weights:  # weights for each net length:  221142
placedb.row_height: # example: 12.0
placedb.site_width: # example: 1.0
placedb.bin_size_x: # example: 20.8828125
placedb.bin_size_y: # example: 20.859375
placedb.num_bins_x: # example: 512
placedb.num_bins_y: # example: 512
placedb.num_movable_pins: # example: 916434
placedb.total_movable_node_area: # total movable cell area, example: 86450368.0
placedb.total_space_area: # total placeable space area excluding fixed cells, example: 114190560.0
placedb.total_filler_node_area: # example: 27740192.0
placedb.total_space_area: # total placeable space area excluding fixed cells, example: 114190560.0
placedb.num_routing_grids_y: # example: 512
placedb.unit_horizontal_capacity: # per unit distance, projected to one layer, example: 1.5625
placedb.unit_vertical_capacity: # per unit distance, projected to one layer, example: 1.45
placedb.total_movable_macro_area: # exaxmple: 49108776.0
placedb.total_movable_cell_area: # example: 37341592.0

## Instructions
- Your objective is to improve given Dreamplace Macro Init to achieve better HPWL performance.
- Comprehend Background, Current Dreamplace Macro Init pytorch implementation snippet. You have to leverage your placement knowledge.
- You can leverage Available placedb objects to extract useful information. Regardining objects with different length, it can not be used directly. It's recommended to extract useful statistical information that can be used in Macro Init.
- Choose the most possible direction to improve the Dreamplace Macro Init!
- To Improve the performance, you can delete or modify some parts of original Macro Init with new implementations.
- End with Output Format!
- For Output, Do not import torch. Do not introduce extra class or undefined class, def or object.
- If you introduce new object, class or define, you should implement it from scratch and merge it into Output
- Ensure all variables are on the same device, i.e., cuda:0

## Output Format
```python

'''Key improvement points summary
## Key improvements from original Dreamplace Macro Init
[Your summary here]
'''

class BasicPlace(nn.Module):
    def __init__(self, params, placedb, timer):
        torch.manual_seed(params.random_seed)
        super(BasicPlace, self).__init__()
        tt = time.time()
        self.init_pos = np.zeros(placedb.num_nodes * 2, dtype=placedb.dtype)
        [Your improved Macro Init implementation here]

```

Your task is to provide a thoroughly improved optimizer that follows Instructions.
Please approach this task with utmost attention to detail and accuracy.

</pre>

# How to run results
Use extract_results.py to obtain benchmark results. Set root_path to point to your total_best_configs directory.
```
# MMS baseline
python extract_results.py --benchmark mms

# MMS ours
python extract_results.py --benchmark mms_ours --root_path <your_root_path>

# ISPD2005 baseline
python extract_results.py --benchmark ispd2005free

# ISPD2005 ours
python extract_results.py --benchmark ispd2005free_ours --root_path <your_root_path>

# ISPD2019 baseline
python extract_results.py --benchmark ispd2019

# ISPD2019 ours
python extract_results.py --benchmark ispd2019_ours --root_path <your_root_path>
```

# Notes
Due to size constraints, we have omitted the generated thoughts, all discovered algorithms, and embeddings files from this anonymous GitHub repository. These files will be provided in the future open-source GitHub version.