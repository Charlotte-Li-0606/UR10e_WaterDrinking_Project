# `my_human_face` Asset

The project includes a complete, textured `person_standing` visual from the
[OSRF Gazebo Models repository](https://github.com/osrf/gazebo_models/tree/master/person_standing):

```text
my_human_face/
├── meshes/
│   └── standing.dae
├── materials/
│   └── textures/
│       ├── young_lightskinned_male_diffuse.png
│       ├── green_eye.png
│       ├── eyebrow001.png
│       └── ...
├── model.config
└── model.sdf
```

`standing.dae` references the PNG files through `../materials/textures/`.
Keep that directory structure intact or the human will appear untextured.

The mesh is visual-only for RGB-D observation. Keep all safety geometry in
`models/human_collision_proxy`; do not add mesh collision to this model.

The upstream asset is licensed under CC BY 3.0; retain appropriate attribution
if distributing it outside this project. See the
[upstream license](https://github.com/osrf/gazebo_models/blob/master/LICENSE).
