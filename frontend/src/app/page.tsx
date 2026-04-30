"use client";

import Link from "next/link";
import { motion } from "framer-motion";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-50 flex flex-col items-center justify-center">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="max-w-2xl w-full px-6 text-center space-y-6"
      >
        <h1 className="text-4xl md:text-5xl font-semibold tracking-tight">
          protoRAG
        </h1>
        <p className="text-neutral-500 dark:text-neutral-400">
          Local-first, privacy-preserving RAG workspace with dynamic LLM configuration.
        </p>
        <motion.div
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          className="flex items-center justify-center mt-8"
        >
          <Link
            href="/chat"
            className="rounded-full bg-neutral-900 text-neutral-50 px-8 py-3 text-sm font-medium shadow-sm hover:bg-neutral-800 dark:bg-neutral-100 dark:text-neutral-900 dark:hover:bg-white transition-colors"
          >
            let&apos;s protoRAG
          </Link>
        </motion.div>
      </motion.div>
    </main>
  );
}
