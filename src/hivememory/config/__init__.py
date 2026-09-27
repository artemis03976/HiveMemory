"""HiveMemory 配置：按子系统与高聚合组件组织的配置段模型，以及应用根配置与加载。

- 配置段模块（``shared`` / ``patchouli`` / ``gateway`` / ``alice`` /
  ``memory_compiler`` / ``attachments`` / ``workspace`` / ``runtime`` /
  ``passive`` / ``access``）：纯配置模型，各层按需导入自己的配置段；
- ``app``：根配置 ``HiveMemoryConfig`` 与加载函数，只供组合根与入口使用。

本包位于最底层，只依赖 pydantic、``hivememory.core`` 常量与 ``hivememory.i18n``；
包初始化不做任何导出，避免经包名绕过上述使用约束。
"""
