// Input schemas for Claude Code's built-in tools, as a proxy model needs them
// to call those tools. The mod API lists each tool's name and description but
// not its schema, so the common ones are written out here; any other tool
// (MCP, plugin, newer built-ins) is not offered.

type Schema = Record<string, unknown>

const str = (description: string): Schema => ({ type: 'string', description })
const num = (description: string): Schema => ({ type: 'number', description })
const bool = (description: string): Schema => ({ type: 'boolean', description })

const object = (properties: Record<string, Schema>, required: string[]): Schema => ({
  type: 'object',
  properties,
  required,
  additionalProperties: false,
})

export const KNOWN_SCHEMAS: Record<string, Schema> = {
  Bash: object(
    {
      command: str('The shell command to run'),
      description: str('A short description of what the command does'),
      timeout: num('Timeout in milliseconds (max 600000)'),
      run_in_background: bool('Run the command in the background'),
    },
    ['command'],
  ),
  Read: object(
    {
      file_path: str('Absolute path of the file to read'),
      offset: num('Line number to start reading from'),
      limit: num('Number of lines to read'),
    },
    ['file_path'],
  ),
  Write: object(
    {
      file_path: str('Absolute path of the file to write'),
      content: str('The full content to write'),
    },
    ['file_path', 'content'],
  ),
  Edit: object(
    {
      file_path: str('Absolute path of the file to edit'),
      old_string: str('The exact text to replace; must be unique in the file unless replace_all is true'),
      new_string: str('The text to replace it with'),
      replace_all: bool('Replace every occurrence of old_string'),
    },
    ['file_path', 'old_string', 'new_string'],
  ),
  Glob: object(
    {
      pattern: str('Glob pattern such as "**/*.ts"'),
      path: str('Directory to search in; defaults to the working directory'),
    },
    ['pattern'],
  ),
  Grep: object(
    {
      pattern: str('Regular expression to search for'),
      path: str('File or directory to search in'),
      glob: str('Glob filter for files, e.g. "*.js"'),
      output_mode: { type: 'string', enum: ['content', 'files_with_matches', 'count'], description: 'What to return' },
      '-i': bool('Case insensitive search'),
      '-n': bool('Show line numbers (content mode)'),
    },
    ['pattern'],
  ),
  WebFetch: object(
    {
      url: str('The URL to fetch'),
      prompt: str('What to extract from the page'),
    },
    ['url', 'prompt'],
  ),
  WebSearch: object({ query: str('The search query') }, ['query']),
  Agent: object(
    {
      description: str('A short (3-5 word) description of the task'),
      prompt: str('The task for the agent'),
      subagent_type: str('The agent type to use'),
    },
    ['description', 'prompt'],
  ),
}
