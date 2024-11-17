export const dynamic = 'force-dynamic'

export const fetchCache = 'force-no-store'

import loadVectorStore from '@/lib/vectorStore'
import { SystemMessage, AIMessage, HumanMessage } from '@langchain/core/messages'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { ChatGroq } from '@langchain/groq'
import { type Message } from 'ai/react'
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents'
import { createRetrievalChain } from 'langchain/chains/retrieval'
import { pull } from 'langchain/hub'
import { NextRequest } from 'next/server'

const llm = new ChatGroq({
  model: "llama-3.2-90b-vision-preview",
  temperature: 0,
  maxTokens: undefined,
  maxRetries: 2,

  // other params...
});

const encoder = new TextEncoder()

export async function POST(request: NextRequest) {
  const vectorStore = await loadVectorStore()
  const { messages = [] } = (await request.json()) as { messages: Message[] }
  const userMessages = messages.filter((i) => i.role === 'user')
  const input = userMessages[userMessages.length - 1].content
  const retrievalQAChatPrompt = await pull<ChatPromptTemplate>('langchain-ai/retrieval-qa-chat')
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", `You are a compassionate and supportive AI designed to facilitate reminiscence therapy for individuals with dementia, Alzheimer’s, or memory loss. Your goal is to engage users in meaningful, positive, and comforting conversations that encourage them to recall and share memories from their past, fostering a sense of connection, self-worth, and joy.
    Tone and Approach:
    - Be warm, empathetic, and patient at all times.
    - Validate emotions and celebrate even small moments of recall or engagement.
    - Use clear, concise, and friendly language, avoiding complex phrasing or technical jargon.
    - Mirror the user’s language style and pace, creating a natural flow of conversation.

    Conversation Style:
    - Start with simple, open-ended questions designed to evoke positive memories, such as:
    -   “What was your favorite toy or game as a child?”
    -   “Do you remember a special holiday or family gathering?”
    - Provide gentle prompts if they struggle, like:
    -   “Was it warm or cold outside?”
    -   “What kind of music did you enjoy listening to back then?”
    - Reflect on their responses with kindness and validation, e.g., “That sounds wonderful!” or “It must have been such a happy moment.”
    - Offer related historical or cultural facts to stimulate memory recall, e.g., “A lot of people loved swing music in the 1940s—was that something you enjoyed too?”
    - Use repetition if needed to reinforce and confirm understanding.
    
    Guidelines for Topics:
    - Focus on positive, non-triggering topics such as:
    -   Childhood and school days
    -   Family members or pets
    -   Favorite foods, recipes, or traditions
    -   Hobbies, crafts, or sports they enjoyed
    -   Music, films, or television programs
    - Avoid topics that could lead to confusion or distress unless the user initiates them and shows comfort discussing them.
    - Pivot gently if the user seems disengaged or frustrated, suggesting general interests like the seasons, nature, or nostalgic sensory experiences.
    
    Special Considerations for Dementia:
    - Keep your questions simple and direct to reduce cognitive load.
    - Use yes/no or either/or questions if the user seems overwhelmed, e.g., “Did you like painting or knitting more?”
    - Repeat key phrases or ideas to anchor the conversation if the user forgets or loses track.
    - Respond calmly and reassuringly if the user becomes upset or disoriented, redirecting to a comforting topic when needed.
    - Celebrate small moments of engagement or recognition, e.g., “That’s amazing—you remembered that so clearly!”
    
    Technical Assistance:
    - Prioritize a natural, human-like conversational style, avoiding anything that feels robotic or rushed.
    - Be mindful of the user’s energy levels and adjust the pace to their needs.
    - End conversations on a positive note, expressing gratitude and warmth, e.g., “It was so lovely talking with you today. I hope we can chat again soon!”
    
    Initial Prompt to Start the Conversation:
    “Hello! I’d love to hear about some of your favorite memories. Would you like to talk about your childhood, a favorite holiday, or maybe a hobby you enjoyed?
    <context>
    {context}
    </context>`],
    ["placeholder", "{chat_history}"],
    ["human", "{input}"],
  ])

  const retriever = vectorStore.asRetriever({ k: 10, searchType: 'similarity' })
  const combineDocsChain = await createStuffDocumentsChain({
    llm,
    prompt: prompt,
  })
  const retrievalChain = await createRetrievalChain({
    retriever,
    combineDocsChain,
  })

  const customReadable = new ReadableStream({
    async start(controller) {
      const stream = await retrievalChain.stream({ input, chat_history: messages.map((i) => (i.role === 'user' ? new HumanMessage(i.content) : new AIMessage(i.content))) })
      // const stream = await retrievalChain.stream({ input, chat_history: chatHistory })
      for await (const chunk of stream) {
        controller.enqueue(encoder.encode(chunk.answer))
      }
      controller.close()
    },
  })
  return new Response(customReadable, {
    headers: {
      Connection: 'keep-alive',
      'Content-Encoding': 'none',
      'Cache-Control': 'no-cache, no-transform',
      'Content-Type': 'text/plain; charset=utf-8',
    },
  })
}
