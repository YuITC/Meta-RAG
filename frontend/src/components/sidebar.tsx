"use client";

import { useEffect } from "react";
import { useStore } from "@/lib/store";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { FileUpload } from "./file-upload";
import { HuggingFacePapers } from "./huggingface-papers";
import { X } from "lucide-react";

export function Sidebar() {
  const open = useStore((s) => s.sidebarOpen);
  const toggle = useStore((s) => s.toggleSidebar);
  const fetchDocuments = useStore((s) => s.fetchDocuments);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  return (
    <>
      {/* Overlay for mobile */}
      {open && (
        <div
          className="fixed inset-0 z-30 bg-black/20 lg:hidden"
          onClick={toggle}
        />
      )}

      <aside
        className={`fixed top-12 left-0 z-40 flex h-[calc(100vh-3rem)] w-80 flex-col border-r bg-background transition-transform duration-200 lg:static lg:z-0 ${
          open ? "translate-x-0" : "-translate-x-full lg:hidden"
        }`}
      >
        <div className="flex h-10 shrink-0 items-center justify-between border-b px-4">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Data Sources
          </span>
          <button
            onClick={toggle}
            className="text-muted-foreground hover:text-foreground lg:hidden"
          >
            <X size={14} />
          </button>
        </div>

        <ScrollArea className="flex-1 px-4 py-3">
          <div className="space-y-5">
            <FileUpload />
            <Separator />
            <HuggingFacePapers />
          </div>
        </ScrollArea>
      </aside>
    </>
  );
}
