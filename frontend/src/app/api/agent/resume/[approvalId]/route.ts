import { NextResponse } from "next/server"

const API_URL =
  process.env.KNOWLEDGEOPS_API_URL ??
  "http://127.0.0.1:8002"


export async function POST(
  _request: Request,
  context: {
    params: Promise<{
      approvalId: string
    }>
  },
) {
  const { approvalId } =
    await context.params

  try {
    const response = await fetch(
      `${API_URL}/agent/resume/${approvalId}`,
      {
        method: "POST",
        cache: "no-store",
      },
    )

    const data =
      await response.json()

    return NextResponse.json(
      data,
      {
        status: response.status,
      },
    )
  } catch {
    return NextResponse.json(
      {
        detail:
          "KnowledgeOps backend is unavailable",
      },
      {
        status: 503,
      },
    )
  }
}