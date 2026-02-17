# pymd 分子动力学模拟器

一个用Python编写的模块化分子动力学(MD)模拟框架。

## 特性

- **多种势能函数**: Lennard-Jones, Morse, EAM
- **灵活的边界条件**: 周期性、开放、混合
- **多种恒温器**: NVE, Berendsen, Nose-Hoover
- **晶格生成器**: FCC, BCC, SC
- **YAML配置**: 通过配置文件设置模拟

## 安装

```bash
pip install numpy
# 可选
pip install pyyaml
```

### 使用Conda

```bash
conda env create -f environment.yml
conda activate pymd
```

## 快速开始

```python
from pymd.builder import SystemBuilder
from pymd.core import Units
from pymd.force import ForceCalculator, NumericalBackend
from pymd.integrator import VelocityVerlet
from pymd.potential import LennardJonesPotential
from pymd.simulator import Simulator
from pymd.thermostat import NoThermostat

# 构建系统
system = (
    SystemBuilder()
    .element("Ar", mass=1.0)
    .fcc_lattice(nx=2, ny=2, nz=2, a=1.5)
    .temperature(0.8)
    .units(Units.LJ())
    .build()
)

# 设置势能和力计算
potential = LennardJonesPotential(epsilon=1.0, sigma=1.0, cutoff=2.5)
force_calc = ForceCalculator(potential, NumericalBackend())

# 运行模拟
sim = Simulator(
    system=system,
    integrator=VelocityVerlet(dt=0.005),
    force_calculator=force_calc,
    thermostat=NoThermostat(),
)
sim.run(num_steps=100)
```

## 运行示例

```bash
python examples/quick_test.py
```

## 测试

```bash
python -m pytest tests/unit/ -q
# 148 tests passing
```

## 文档

- [用户手册](docs/用户手册.md)
- [开发者指南](docs/开发者指南.md)

## 许可证

MIT

## 作者

本软件由 **Prof. Bin Shan (单斌教授)** 开发。
如有问题或建议，请联系：bshan@mail.hust.edu.cn

