import assert from 'node:assert/strict'
import { test } from 'node:test'

import { configFromEnv } from '../lib/config.js'

test('credential comes only from the dsh variable, not from another harness in the same environment', () => {
  const config = configFromEnv({
    HIVEMEMORY_CLAUDE_CODE_TOKEN: 'claude-code-token',
    HIVEMEMORY_TOKEN: 'shared-token',
    HIVEMEMORY_DSH_TOKEN: 'dsh-token',
  })

  assert.equal(config.token, 'dsh-token')
})
