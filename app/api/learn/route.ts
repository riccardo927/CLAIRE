export const dynamic = 'force-dynamic'

export const fetchCache = 'force-no-store'

import loadVectorStore from '@/lib/vectorStore'
import { Document } from '@langchain/core/documents'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'


export async function POST(request: Request) {
  const { message: text } = await request.json()
  if (!text) return new Response(null, { status: 400 })
  const document = new Document({ pageContent: text })
  // const vectorStore = await loadVectorStore()
  // await vectorStore.addDocuments([document])
  // return new Response()

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  })
  let splitDocs = await textSplitter.splitDocuments([document])
  const vectorStore = await loadVectorStore()
  splitDocs = splitDocs.map((i) => ({ ...i, pageContent: i.pageContent.replace(/\0/g, '') }))
  await vectorStore.addDocuments(splitDocs)
  return new Response()
}


