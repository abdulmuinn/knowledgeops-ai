import { NextResponse } from "next/server"

const API_URL =
  process.env.KNOWLEDGEOPS_API_URL ??
  "http://127.0.0.1:8000"


export async function POST(
  request: Request,
) {
  const body = await request.json()

  try {
    const response = await fetch(
      `${API_URL}/agent/run`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        cache: "no-store",
      },
    )

    const data = await response.json()

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