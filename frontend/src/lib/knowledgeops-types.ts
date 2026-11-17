export type SourceResponse = {
  citation_id: string
  filename: string
  page_number: number | null
  chunk_index: number
}

export type ApprovalResponse = {
  approval_id: string
  status: string
  tool_name: string
  arguments: Record<string, unknown>
  reason: string
}

export type ToolResultResponse = {
  status: string
  output: unknown | null
  error: string | null
}

export type AgentRunResponse = {
  status:
    | "rag_completed"
    | "tool_completed"
    | "approval_required"

  answer: string | null
  sources: SourceResponse[]

  tool_name: string | null
  tool_arguments: Record<string, unknown> | null

  approval: ApprovalResponse | null
  tool_result: ToolResultResponse | null
}