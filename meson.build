project(
  'bottleneck',
  'c',
  # https://mesonbuild.com/meson-python/reference/meson-compatibility.html#cmdoption-arg-1.6.0
  meson_version: '>=1.6.0',
)

# detect missing dependencies early
py = import('python').find_installation(pure: false)

helpers = 'build_helpers'

npy_inc_str = run_command(py,
  [helpers / 'get_numpy_include_dir.py'],
  check: true,
).stdout().strip()
npy_inc = include_directories(npy_inc_str)

npy_nodeprapi = [
  # keep in sync with pyproject.toml
  '-DNPY_NO_DEPRECATED_API=NPY_1_21_API_VERSION',
]

npy_dep = declare_dependency(
  include_directories: npy_inc,
  compile_args: npy_nodeprapi,
)

bn_config_h = custom_target(
  'bn_config',
  output: 'bn_config.h',
  command: [find_program(helpers / 'config.py'), '-o', '@OUTDIR@'],
)

cgen = generator(
  find_program(helpers / 'template.py'),
  arguments: ['@INPUT@', '-o', '@OUTPUT@'],
  output: '@BASENAME@',
)

bn = 'bottleneck'
src = bn / 'src'
bn_dep = declare_dependency(
  dependencies: npy_dep,
  compile_args: npy_nodeprapi,
  include_directories: [src, npy_inc],
)

py.extension_module(
  'reduce',
  [
    bn_config_h,
    cgen.process(src / 'reduce.c.template'),
  ],
  subdir: 'bottleneck',
  install: true,
  dependencies : [npy_dep, bn_dep],
  c_args: npy_nodeprapi,
  limited_api: py.language_version(),
)
py.extension_module(
  'move',
  [
    bn_config_h,
    cgen.process(src / 'move.c.template'),
    src / 'move_median/move_median.c',
  ],
  subdir: 'bottleneck',
  install: true,
  dependencies : [npy_dep, bn_dep],
  c_args: npy_nodeprapi,
  limited_api: py.language_version(),
)
py.extension_module(
  'nonreduce',
  [
    bn_config_h,
    cgen.process(src / 'nonreduce.c.template'),
  ],
  subdir: 'bottleneck',
  install: true,
  dependencies : [npy_dep, bn_dep],
  c_args: npy_nodeprapi,
  limited_api: py.language_version(),
)
py.extension_module(
  'nonreduce_axis',
  [
    bn_config_h,
    cgen.process(src / 'nonreduce_axis.c.template'),
  ],
  subdir: 'bottleneck',
  install: true,
  dependencies : [npy_dep, bn_dep],
  c_args: npy_nodeprapi,
  limited_api: py.language_version(),
)


py.install_sources(
  bn / '__init__.py',
  bn / '_pytesttester.py',
  bn / 'conftest.py',
  # bn / '_version.py',
  subdir: 'bottleneck',
)

benchmark = bn / 'benchmark'
py.install_sources(
  benchmark / '__init__.py',
  benchmark / 'autotimeit.py',
  benchmark / 'bench_detailed.py',
  benchmark / 'bench.py',
  subdir: 'bottleneck/benchmark',
)

slow = bn / 'slow'
py.install_sources(
  slow / '__init__.py',
  slow / 'move.py',
  slow / 'nonreduce_axis.py',
  slow / 'nonreduce.py',
  slow / 'reduce.py',
  subdir: 'bottleneck/slow',
)

tests = bn / 'tests'
py.install_sources(
  tests / '__init__.py',
  tests / 'common.py',
  tests / 'input_modification_test.py',
  tests / 'list_input_test.py',
  tests / 'memory_test.py',
  tests / 'move_test.py',
  tests / 'nonreduce_axis_test.py',
  tests / 'nonreduce_test.py',
  tests / 'reduce_test.py',
  tests / 'scalar_input_test.py',
  tests / 'util.py',
  subdir: 'bottleneck/tests',
)
