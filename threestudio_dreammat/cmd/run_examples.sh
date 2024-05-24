python launch.py --config configs/dreammat.yaml --train --gradio --gpu 0  system.prompt_processor.prompt="A red apple" system.geometry.shape_init=mesh:load/shapes/objs/apple.obj trainer.max_steps=3000 system.geometry.shape_init_params=0.7 data.blender_generate=true
python launch.py --config configs/dreammat.yaml --train --gradio --gpu 0  system.prompt_processor.prompt="A strawberry" system.geometry.shape_init=mesh:load/shapes/objs/strawberry.obj trainer.max_steps=3000 system.geometry.shape_init_params=0.8 data.blender_generate=true
python launch.py --config configs/dreammat.yaml --train --gradio --gpu 0  system.prompt_processor.prompt="A cute striped kitten" system.geometry.shape_init=mesh:load/shapes/objs/cat.obj trainer.max_steps=4000 system.geometry.shape_init_params=0.85 data.blender_generate=true
python launch.py --config configs/dreammat.yaml --train --gradio --gpu 0  system.prompt_processor.prompt="A turtle" system.geometry.shape_init=mesh:load/shapes/objs/turtle.obj trainer.max_steps=3000 system.geometry.shape_init_params=1.0 data.blender_generate=true
python launch.py --config configs/dreammat.yaml --train --gradio --gpu 0  system.prompt_processor.prompt="A brown basketball" system.geometry.shape_init=mesh:load/shapes/objs/basketball.obj trainer.max_steps=3000 system.geometry.shape_init_params=0.6 data.blender_generate=true
python launch.py --config configs/dreammat.yaml --train --gradio --gpu 0  system.prompt_processor.prompt="the earth" system.geometry.shape_init=mesh:load/shapes/objs/sphere.obj trainer.max_steps=4000 system.geometry.shape_init_params=0.6 system.guidance.cond_scale=1.02 data.blender_generate=true
python launch.py --config configs/dreammat.yaml --train --gradio --gpu 0  system.prompt_processor.prompt="A natural grey rabbit" system.geometry.shape_init=mesh:load/shapes/objs/rabbit.obj trainer.max_steps=4000 system.geometry.shape_init_params=1.0 data.blender_generate=true