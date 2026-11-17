"use client"

import { useState } from "react"
import {
  CheckCircle2,
  FileText,
  Inbox,
  Search,
  Send,
  ShieldCheck,
} from "lucide-react"

import {
  ThemeToggle,
} from "@/components/theme-toggle"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Textarea } from "@/components/ui/textarea"

import type { AgentRunResponse } from "@/lib/knowledgeops-types"


export function KnowledgeOpsShell() {
  const [message, setMessage] = useState("")

  const [result, setResult] =
    useState<AgentRunResponse | null>(null)

  const [loading, setLoading] =
    useState(false)

  const [error, setError] =
    useState<string | null>(null)

  const [
    approvalAction,
    setApprovalAction,
  ] = useState<
    "approve" | "reject" | null
  >(null)

  const [
    approvalDecision,
    setApprovalDecision,
  ] = useState<
    "rejected" | null
  >(null)


  async function submitRequest(
    requestMessage = message,
  ) {
    const trimmed =
      requestMessage.trim()

    if (!trimmed || loading) {
      return
    }

    setMessage(trimmed)
    setLoading(true)
    setError(null)
    setResult(null)
    setApprovalDecision(null)
    setApprovalAction(null)

    try {
      const response = await fetch(
        "/api/agent/run",
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify({
            message: trimmed,
          }),
        },
      )

      const data =
        await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail ??
            "Request failed",
        )
      }

      setResult(
        data as AgentRunResponse,
      )
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Request failed",
      )
    } finally {
      setLoading(false)
    }
  }


  async function approvePendingRequest() {
    const approvalId =
      result?.approval?.approval_id

    if (
      !approvalId ||
      approvalAction
    ) {
      return
    }

    setApprovalAction("approve")
    setError(null)

    try {
      const approveResponse =
        await fetch(
          `/api/approvals/${approvalId}/approve`,
          {
            method: "POST",
          },
        )

      const approveData =
        await approveResponse.json()

      if (!approveResponse.ok) {
        throw new Error(
          approveData.detail ??
            "Approval failed",
        )
      }

      const resumeResponse =
        await fetch(
          `/api/agent/resume/${approvalId}`,
          {
            method: "POST",
          },
        )

      const resumeData =
        await resumeResponse.json()

      if (!resumeResponse.ok) {
        throw new Error(
          resumeData.detail ??
            "Tool execution failed",
        )
      }

      setApprovalDecision(null)

      setResult(
        resumeData as AgentRunResponse,
      )
    } catch (approvalError) {
      setError(
        approvalError instanceof Error
          ? approvalError.message
          : "Approval failed",
      )
    } finally {
      setApprovalAction(null)
    }
  }


  async function rejectPendingRequest() {
    const approvalId =
      result?.approval?.approval_id

    if (
      !approvalId ||
      approvalAction
    ) {
      return
    }

    setApprovalAction("reject")
    setError(null)

    try {
      const response =
        await fetch(
          `/api/approvals/${approvalId}/reject`,
          {
            method: "POST",
          },
        )

      const data =
        await response.json()

      if (!response.ok) {
        throw new Error(
          data.detail ??
            "Rejection failed",
        )
      }

      setApprovalDecision(
        "rejected",
      )
    } catch (approvalError) {
      setError(
        approvalError instanceof Error
          ? approvalError.message
          : "Rejection failed",
      )
    } finally {
      setApprovalAction(null)
    }
  }


  const pendingApprovalCount =
    result?.status ===
      "approval_required" &&
    approvalDecision !==
      "rejected"
      ? 1
      : 0


  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="flex h-14 items-center justify-between border-b px-5">
        <div className="flex items-center gap-3">
          <div className="flex size-8 items-center justify-center border bg-muted/40 text-xs font-semibold">
            KO
          </div>

          <div>
            <p className="text-sm font-semibold leading-none">
              KnowledgeOps
            </p>

            <p className="mt-1 text-xs text-muted-foreground">
              Internal operations assistant
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <ThemeToggle />
        
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <CheckCircle2 className="size-3.5 text-emerald-600" />
            API healthy
          </div>
        </div>
      </header>


      <div className="grid min-h-[calc(100vh-3.5rem)] grid-cols-[220px_minmax(0,1fr)_300px]">
        {/* LEFT SIDEBAR */}
        <aside className="border-r">
          <div className="p-4">
            <p className="mb-3 text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
              Workspace
            </p>

            <nav className="space-y-1">
              <Button
                variant="secondary"
                className="w-full justify-start"
              >
                <Search className="size-4" />
                Ask
              </Button>

              <Button
                variant="ghost"
                className="w-full justify-between"
              >
                <span className="flex items-center gap-2">
                  <Inbox className="size-4" />
                  Approvals
                </span>

                <Badge
                  variant="outline"
                  className="rounded-sm px-1.5"
                >
                  {
                    pendingApprovalCount
                  }
                </Badge>
              </Button>
            </nav>
          </div>

          <Separator />

          <div className="p-4">
            <p className="mb-3 text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
              System
            </p>

            <div className="space-y-3 text-xs">
              <SystemStatus
                label="RAG"
                value="Ready"
              />

              <SystemStatus
                label="n8n"
                value="Connected"
              />

              <SystemStatus
                label="PostgreSQL"
                value="Connected"
              />
            </div>
          </div>
        </aside>


        {/* MAIN WORKSPACE */}
        <section className="flex min-w-0 flex-col">
          <div className="border-b px-6 py-5">
            <div className="max-w-3xl">
              <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
                Operations workspace
              </p>

              <h1 className="mt-2 text-xl font-semibold tracking-tight">
                Ask a policy question
                or request an action
              </h1>

              <p className="mt-1 max-w-2xl text-sm leading-6 text-muted-foreground">
                Responses are grounded
                in internal knowledge.
                State-changing actions
                require human approval
                before execution.
              </p>
            </div>
          </div>


          {/* RESPONSE AREA */}
          <div className="flex flex-1 items-center justify-center px-6 py-12">
            {result?.status ===
              "rag_completed" &&
            result.answer ? (
              <div className="w-full max-w-2xl">
                <div className="border p-6">
                  <p className="mb-3 text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
                    KnowledgeOps
                  </p>

                  <p className="text-sm leading-7">
                    {result.answer}
                  </p>
                </div>
              </div>
            ) : result?.status ===
              "tool_completed" ? (
              <ToolCompletedResult
                result={result}
              />
            ) : result?.status ===
              "approval_required" ? (
              <div className="w-full max-w-2xl">
                <div className="border p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
                        Action request
                      </p>

                      <h2 className="mt-2 text-base font-semibold">
                        Human approval
                        required
                      </h2>
                    </div>

                    <Badge
                      variant="outline"
                      className="rounded-sm"
                    >
                      {approvalDecision ===
                      "rejected"
                        ? "Rejected"
                        : "Pending"}
                    </Badge>
                  </div>

                  <Separator className="my-5" />

                  <div className="space-y-4 text-sm">
                    <div>
                      <p className="text-xs text-muted-foreground">
                        Tool
                      </p>

                      <p className="mt-1 font-medium">
                        {result.tool_name ??
                          "Unknown"}
                      </p>
                    </div>


                    {result.tool_arguments && (
                      <div className="grid grid-cols-2 gap-4">
                        {Object.entries(
                          result.tool_arguments,
                        ).map(
                          ([key, value]) => (
                            <div key={key}>
                              <p className="text-xs capitalize text-muted-foreground">
                                {key.replaceAll(
                                  "_",
                                  " ",
                                )}
                              </p>

                              <p className="mt-1 font-medium">
                                {String(
                                  value,
                                )}
                              </p>
                            </div>
                          ),
                        )}
                      </div>
                    )}


                    {result.approval?.reason && (
                      <div>
                        <p className="text-xs text-muted-foreground">
                          Reason
                        </p>

                        <p className="mt-1 leading-6">
                          {
                            result
                              .approval
                              .reason
                          }
                        </p>
                      </div>
                    )}
                  </div>


                  {error && (
                    <div className="mt-5 border border-destructive/30 p-3">
                      <p className="text-xs font-medium text-destructive">
                        {error}
                      </p>
                    </div>
                  )}


                  {approvalDecision ===
                  "rejected" ? (
                    <div className="mt-6 border-t pt-5">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <p className="text-sm font-medium">
                            Request
                            rejected
                          </p>

                          <p className="mt-1 text-xs text-muted-foreground">
                            The tool was
                            not executed.
                          </p>
                        </div>

                        <Badge
                          variant="outline"
                          className="rounded-sm"
                        >
                          Rejected
                        </Badge>
                      </div>
                    </div>
                  ) : (
                    <div className="mt-6 flex items-center justify-end gap-2 border-t pt-5">
                      <Button
                        variant="outline"
                        disabled={
                          approvalAction !==
                          null
                        }
                        onClick={() =>
                          void rejectPendingRequest()
                        }
                      >
                        {approvalAction ===
                        "reject"
                          ? "Rejecting..."
                          : "Reject"}
                      </Button>

                      <Button
                        disabled={
                          approvalAction !==
                          null
                        }
                        onClick={() =>
                          void approvePendingRequest()
                        }
                      >
                        {approvalAction ===
                        "approve"
                          ? "Approving..."
                          : "Approve"}
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            ) : error ? (
              <div className="w-full max-w-2xl border border-destructive/30 p-5">
                <p className="text-sm font-medium">
                  Request failed
                </p>

                <p className="mt-1 text-sm text-muted-foreground">
                  {error}
                </p>
              </div>
            ) : loading ? (
              <p className="text-sm text-muted-foreground">
                Processing request…
              </p>
            ) : (
              <div className="w-full max-w-2xl">
                <div className="border border-dashed p-8">
                  <div className="mb-4 flex size-9 items-center justify-center border bg-muted/40">
                    <ShieldCheck className="size-4" />
                  </div>

                  <h2 className="text-sm font-medium">
                    Start with a request
                  </h2>

                  <p className="mt-1 max-w-md text-sm leading-6 text-muted-foreground">
                    Ask about internal
                    policies, or request
                    an operational action
                    such as production
                    access.
                  </p>

                  <div className="mt-5 grid gap-2 sm:grid-cols-2">
                    <ExamplePrompt
                      disabled={
                        loading
                      }
                      onClick={() =>
                        void submitRequest(
                          "Who must approve standard employee access to production systems?",
                        )
                      }
                    >
                      Who must approve
                      production access?
                    </ExamplePrompt>

                    <ExamplePrompt
                      disabled={
                        loading
                      }
                      onClick={() =>
                        void submitRequest(
                          "Create a read-only production access request for me.",
                        )
                      }
                    >
                      Create a read-only
                      production access
                      request.
                    </ExamplePrompt>
                  </div>
                </div>
              </div>
            )}
          </div>


          {/* INPUT */}
          <div className="border-t p-4">
            <div className="mx-auto flex max-w-3xl items-end gap-2">
              <Textarea
                value={message}
                onChange={(event) =>
                  setMessage(
                    event.target.value,
                  )
                }
                onKeyDown={(event) => {
                  if (
                    event.key ===
                      "Enter" &&
                    !event.shiftKey
                  ) {
                    event.preventDefault()

                    if (
                      !loading &&
                      message.trim()
                    ) {
                      void submitRequest()
                    }
                  }
                }}
                placeholder="Ask about policy or request an operational action..."
                className="min-h-20 resize-none rounded-sm"
              />

              <Button
                size="icon"
                className="shrink-0 rounded-sm"
                aria-label="Send request"
                disabled={
                  loading ||
                  !message.trim()
                }
                onClick={() =>
                  void submitRequest()
                }
              >
                <Send className="size-4" />
              </Button>
            </div>

            <p className="mx-auto mt-2 max-w-3xl text-[11px] text-muted-foreground">
              Enter to send ·
              Shift + Enter for a new
              line · State-changing
              actions require approval.
            </p>
          </div>
        </section>


        {/* CONTEXT PANEL */}
        <aside className="border-l">
          <div className="border-b px-4 py-5">
            <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
              Context
            </p>

            <h2 className="mt-2 text-sm font-medium">
              Sources
            </h2>
          </div>

          <div className="p-4">
            {result?.sources.length ? (
              <div className="space-y-2">
                {result.sources.map(
                  (source) => (
                    <div
                      key={
                        source.citation_id
                      }
                      className="border p-3"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <Badge
                          variant="outline"
                          className="rounded-sm"
                        >
                          {
                            source.citation_id
                          }
                        </Badge>

                        <span className="text-[11px] text-muted-foreground">
                          Chunk{" "}
                          {
                            source.chunk_index
                          }
                        </span>
                      </div>

                      <div className="mt-3 flex items-start gap-2">
                        <FileText className="mt-0.5 size-4 shrink-0 text-muted-foreground" />

                        <div className="min-w-0">
                          <p className="break-words text-sm font-medium">
                            {
                              source.filename
                            }
                          </p>

                          {source.page_number !==
                            null && (
                            <p className="mt-1 text-xs text-muted-foreground">
                              Page{" "}
                              {
                                source.page_number
                              }
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  ),
                )}
              </div>
            ) : (
              <div className="flex min-h-48 flex-col items-center justify-center border border-dashed px-4 text-center">
                <FileText className="mb-3 size-4 text-muted-foreground" />

                <p className="text-sm font-medium">
                  No sources yet
                </p>

                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  Grounded citations
                  will appear here
                  after a knowledge
                  request.
                </p>
              </div>
            )}
          </div>
        </aside>
      </div>
    </main>
  )
}


function ToolCompletedResult({
  result,
}: {
  result: AgentRunResponse
}) {
  const output =
    result.tool_result?.output

  const outputEntries =
    output &&
    typeof output === "object" &&
    !Array.isArray(output)
      ? Object.entries(
          output as Record<
            string,
            unknown
          >,
        )
      : []

  return (
    <div className="w-full max-w-2xl">
      <div className="border p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
              Action completed
            </p>

            <h2 className="mt-2 text-base font-semibold">
              Access request submitted
            </h2>
          </div>

          <Badge
            variant="outline"
            className="rounded-sm"
          >
            {result.tool_result
              ?.status ??
              "completed"}
          </Badge>
        </div>

        <Separator className="my-5" />

        {outputEntries.length >
        0 ? (
          <div className="grid gap-4 sm:grid-cols-2">
            {outputEntries.map(
              ([key, value]) => (
                <div key={key}>
                  <p className="text-xs capitalize text-muted-foreground">
                    {key.replaceAll(
                      "_",
                      " ",
                    )}
                  </p>

                  <p className="mt-1 break-words text-sm font-medium">
                    {String(value)}
                  </p>
                </div>
              ),
            )}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            The action completed
            successfully.
          </p>
        )}

        {result.tool_result
          ?.error && (
          <div className="mt-5 border border-destructive/30 p-3">
            <p className="text-sm text-destructive">
              {
                result.tool_result
                  .error
              }
            </p>
          </div>
        )}
      </div>
    </div>
  )
}


function SystemStatus({
  label,
  value,
}: {
  label: string
  value: string
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground">
        {label}
      </span>

      <span className="flex items-center gap-1.5">
        <span className="size-1.5 bg-emerald-600" />
        {value}
      </span>
    </div>
  )
}


function ExamplePrompt({
  children,
  onClick,
  disabled,
}: {
  children: React.ReactNode
  onClick: () => void
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className="border px-3 py-2.5 text-left text-xs leading-5 text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
    >
      {children}
    </button>
  )
}