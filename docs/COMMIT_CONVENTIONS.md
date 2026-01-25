# Commit 规范

## Commit 原则

1. **按功能提交**：一个完整的功能是一个 commit
2. **相关修改合并**：代码、测试、文档尽量在一个 commit
3. **合理的粒度**：纯代码不超过 500 行

## Commit 消息格式

```
<type>(<scope>): <subject>

<body>

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

## 类型

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档
- `refactor`: 重构
- `test`: 测试
- `chore`: 杂项

## 示例

```
feat(data): 实现数据获取模块

- 实现 MockDataFetcher
- 实现 TushareDataFetcher
- 添加缓存和存储
- 所有测试通过

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```
