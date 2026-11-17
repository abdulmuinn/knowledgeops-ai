"use client"

import {
  Moon,
  Sun,
} from "lucide-react"
import {
  useTheme,
} from "next-themes"

import {
  Button,
} from "@/components/ui/button"


export function ThemeToggle() {
  const {
    resolvedTheme,
    setTheme,
  } = useTheme()


  function toggleTheme() {
    setTheme(
      resolvedTheme === "dark"
        ? "light"
        : "dark",
    )
  }


  return (
    <Button
      variant="ghost"
      size="icon"
      className="size-8 rounded-sm"
      aria-label="Toggle color theme"
      title="Toggle color theme"
      onClick={toggleTheme}
    >
      <Sun className="hidden size-4 dark:block" />
      <Moon className="size-4 dark:hidden" />

      <span className="sr-only">
        Toggle color theme
      </span>
    </Button>
  )
}