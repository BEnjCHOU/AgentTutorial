'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Send, Upload, FileText, Bot, User, Trash2, Settings, BarChart3, Brain, Zap } from 'lucide-react';
import axios, { AxiosError } from 'axios';

// --- Types ---
interface Message {
  role: 'user' | 'agent';
  content: string;
  reasoning_steps?: string[];
  action_steps?: Array<{
    tool_name: string;
    input: Record<string, any>;
    output: string;
  }>;
  evaluation?: {
    overall_quality_score: number;
    relevance: {
      relevance_score: number;
      explanation: string;
      key_points: string[];
    };
    completeness: {
      completeness_score: number;
      explanation: string;
      missing_aspects: string[];
    };
    recommendation: string;
  };
}

type TaskType = 'default' | 'document_analysis' | 'research' | 'calculation' | 'general';

export default function ChatAgent() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'agent', content: 'Hello! I am your AI assistant. Upload a file or ask me anything.' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [files, setFiles] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [updatingFileName, setUpdatingFileName] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [taskType, setTaskType] = useState<TaskType>('default');
  const [evaluateContext, setEvaluateContext] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [useStreaming, setUseStreaming] = useState(true); // Enable streaming by default
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);        // for new uploads
  const updateFileInputRef = useRef<HTMLInputElement>(null);  // for updates

  // Backend URL - use environment variable if set, otherwise default to localhost
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // --- Effects ---
  useEffect(() => {
    fetchFiles();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // --- API Calls ---
  const fetchFiles = async () => {
    try {
      const res = await axios.get(`${API_URL}/files/`);
      setFiles(res.data.files || []);
    } catch (err) {
      console.error("Error fetching files:", err);
    }
  };

  const handleSendMessage = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    // Create a placeholder message for streaming
    const messageId = Date.now();
    const placeholderMessage: Message = {
      role: 'agent',
      content: '',
      reasoning_steps: [],
      action_steps: []
    };
    setMessages(prev => [...prev, placeholderMessage]);

    try {
      if (useStreaming) {
        // Use streaming endpoint
        const response = await fetch(`${API_URL}/ask-stream/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: userMessage,
            task_type: taskType,
            evaluate_context: evaluateContext
          }),
        });

        if (!response.ok) {
          throw new Error('Streaming request failed');
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        if (!reader) {
          throw new Error('No reader available');
        }

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                setMessages(prev => {
                  const updated = [...prev];
                  // Create a new message object to ensure React detects changes
                  const lastMsg = { ...updated[updated.length - 1] };
                  
                  if (data.type === 'reasoning') {
                    // Handle reasoning content
                    const reasoningSteps = [...(lastMsg.reasoning_steps || [])];
                    // Create new step if: no steps yet, backend explicitly signals new step, or state changed
                    const shouldCreateNewStep = (
                      reasoningSteps.length === 0 || 
                      data.new_reasoning === true || 
                      data.state_changed === true
                    );
                    if (shouldCreateNewStep) {
                      // Create a new reasoning step (new chat box)
                      reasoningSteps.push(data.content);
                    } else {
                      // Append to last reasoning step for streaming continuity within same step
                      reasoningSteps[reasoningSteps.length - 1] = reasoningSteps[reasoningSteps.length - 1] + data.content;
                    }
                    
                    // Create new message object with updated reasoning steps
                    updated[updated.length - 1] = {
                      ...lastMsg,
                      reasoning_steps: reasoningSteps
                    };
                  } else if (data.type === 'action') {
                    // Add action step (actions always create a new step, indicated by state_changed)
                    const actionSteps = [...(lastMsg.action_steps || [])];
                    actionSteps.push({
                      tool_name: data.tool_name,
                      input: data.tool_input,
                      output: data.tool_output
                    });
                    
                    // Create new message object with updated action steps
                    updated[updated.length - 1] = {
                      ...lastMsg,
                      action_steps: actionSteps
                    };
                  } else if (data.type === 'answer') {
                    // For answers, if state_changed is true, we might want to treat it as a new answer
                    // For now, we'll append to content, but you could also track answer steps separately
                    const newContent = (lastMsg.content || '') + data.content;
                    updated[updated.length - 1] = {
                      ...lastMsg,
                      content: newContent
                    };
                  } else if (data.type === 'evaluation') {
                    // Add evaluation to the message
                    updated[updated.length - 1] = {
                      ...lastMsg,
                      evaluation: data.evaluation
                    };
                  } else if (data.type === 'final') {
                    // Final event - just mark as complete, content already streamed
                    // No need to update anything, all content was already streamed
                  }
                  
                  return updated;
                });
              } catch (err) {
                console.error('Error parsing SSE data:', err);
              }
            }
          }
        }
      } else {
        // Use non-streaming endpoint
      const res = await axios.post(`${API_URL}/ask/`, { 
        message: userMessage,
        task_type: taskType,
        evaluate_context: evaluateContext
      });
      const agentResponse = res.data.response || "No response received.";
      const message: Message = { 
        role: 'agent', 
        content: agentResponse 
      };
      if (res.data.evaluation) {
        message.evaluation = res.data.evaluation;
      }
        if (res.data.reasoning_steps) {
          message.reasoning_steps = res.data.reasoning_steps;
        }
        if (res.data.action_steps) {
          message.action_steps = res.data.action_steps;
        }
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = message;
          return updated;
        });
      }
    } catch (error) {
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = { 
          role: 'agent', 
          content: "Error: Could not connect to the agent." 
        };
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API_URL}/upload/`, formData);
      // if file exists, alert user to use update instead
      if ('error' in res.data) {
        setMessages(prev => [...prev, { 
          role: 'agent', 
          content: `${res.data.error}`
        }]);
        return
      }
      await fetchFiles();
      setMessages(prev => [...prev, { 
        role: 'agent', 
        content: `✅ Successfully processed file: ${file.name}. You can now ask questions about it.` 
      }]);
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        console.error("Upload failed. Server response data:", error.response.data);
        alert(`Error uploading file: ${error.response.data.message || 'Server rejected the file.'}`);
      } else {
        console.error("An unknown error occurred during upload:", error);
        alert("An unknown error occurred during file upload.");
      }
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // --- Update feature handlers ---
  const startUpdate = (filename: string) => {
    // mark which file will be updated, then open the hidden file input
    setUpdatingFileName(filename);
    updateFileInputRef.current?.click();
  };

  const handleUpdateFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !updatingFileName) {
      setUpdatingFileName(null);
      if (updateFileInputRef.current) updateFileInputRef.current.value = '';
      return;
    }

    setIsUpdating(true);
    const formData = new FormData();
    formData.append("file", file);
    // include the target filename so backend knows which file to replace
    formData.append("target_filename", updatingFileName);

    try {
      const res = await axios.put(`${API_URL}/update/`, formData);
      if ('error' in res.data) {
        setMessages(prev => [...prev, { 
          role: 'agent', 
          content: `${res.data.status}\n${res.data.error}`
        }]);
        return;
      }
      await fetchFiles();
      setMessages(prev => [...prev, { 
        role: 'agent', 
        content: `🔁 Successfully updated ${updatingFileName} with ${file.name}.` 
      }]);
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        console.error("Update failed. Server response data:", error.response.data);
        alert(`Error updating file: ${error.response.data.message || 'Server rejected the file.'}`);
      } else {
        console.error("An unknown error occurred during update:", error);
        alert("An unknown error occurred during file update.");
      }
      setMessages(prev => [...prev, { role: 'agent', content: `❌ Failed to update ${updatingFileName}.` }]);
    } finally {
      setIsUpdating(false);
      setUpdatingFileName(null);
      if (updateFileInputRef.current) updateFileInputRef.current.value = '';
    }
  };

  // --- Delete feature handler ---
  const handleDeleteFile = async (filename: string) => {
    // Confirm before deleting
    if (!window.confirm(`Are you sure you want to delete ${filename}?`)) {
      return;
    }

    setIsDeleting(true);
    try {
      const res = await axios.delete(`${API_URL}/delete/${filename}`);
      if ('error' in res.data) {
        setMessages(prev => [...prev, { 
          role: 'agent', 
          content: `❌ Failed to delete ${filename}: ${res.data.error}`
        }]);
        return;
      }

      await fetchFiles();
      setMessages(prev => [...prev, { 
        role: 'agent', 
        content: `🗑️ Successfully deleted ${filename} from the knowledge base.` 
      }]);
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        console.error("Delete failed. Server response data:", error.response.data);
        alert(`Error deleting file: ${error.response.data.message || 'Server rejected the request.'}`);
      } else {
        console.error("An unknown error occurred during delete:", error);
        alert("An unknown error occurred during file deletion.");
      }
      setMessages(prev => [...prev, { role: 'agent', content: `❌ Failed to delete ${filename}.` }]);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-50 text-gray-900 font-sans">
      
      {/* --- Sidebar (Files) --- */}
      <div className="w-64 bg-white border-r border-gray-200 flex flex-col hidden md:flex">
        <div className="p-4 border-b border-gray-200">
          <h2 className="text-lg font-bold flex items-center gap-2 text-indigo-600">
            <Bot size={24} />
            Agent Brain
          </h2>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
            Knowledge Base
          </h3>
          <div className="space-y-2">
            {files.length === 0 ? (
              <p className="text-sm text-gray-400 italic">No files indexed yet.</p>
            ) : (
              files.map((file, idx) => (
                <div key={idx} className="flex items-center gap-2 text-sm text-gray-700 bg-gray-50 p-2 rounded-md border border-gray-100">
                  <FileText size={16} className="text-blue-500" />
                  <span className="truncate flex-1">{file}</span>

                  {/* Update button */}
                  <button
                    onClick={() => startUpdate(file)}
                    disabled={isUpdating || isUploading || isDeleting}
                    title={`Update ${file}`}
                    className="ml-1 text-xs px-2 py-1 rounded-md bg-yellow-100 text-yellow-800 hover:bg-yellow-200 transition-colors disabled:opacity-50"
                  >
                    Update
                  </button>

                  {/* Delete button */}
                  <button
                    onClick={() => handleDeleteFile(file)}
                    disabled={isDeleting || isUploading || isUpdating}
                    title={`Delete ${file}`}
                    className="ml-1 text-xs px-2 py-1 rounded-md bg-red-100 text-red-800 hover:bg-red-200 transition-colors disabled:opacity-50"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="p-4 border-t border-gray-200">
          {/* hidden file inputs */}
          <input 
            type="file" 
            ref={fileInputRef}
            onChange={handleFileUpload}
            className="hidden"
          />
          <input
            type="file"
            ref={updateFileInputRef}
            onChange={handleUpdateFileChange}
            className="hidden"
          />

          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading || isUpdating || isDeleting}
            className={`w-full flex items-center justify-center gap-2 py-2 px-4 rounded-md text-white transition-colors ${
              isUploading ? 'bg-indigo-300 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'
            }`}
          >
            {isUploading ? (
              <span className="animate-pulse">Uploading...</span>
            ) : (
              <>
                <Upload size={18} />
                Upload Context
              </>
            )}
          </button>
        </div>
      </div>

      {/* --- Main Chat Area --- */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="p-4 bg-white border-b flex justify-between items-center">
          <span className="font-bold text-indigo-600">Agent Chat</span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 text-indigo-600 hover:bg-indigo-50 rounded-md transition-colors"
              title="Settings"
            >
              <Settings size={20} />
            </button>
            <button 
               onClick={() => fileInputRef.current?.click()}
               className="text-sm text-indigo-600 font-medium md:hidden"
            >
              + Upload
            </button>
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="p-4 bg-indigo-50 border-b border-indigo-100">
            <div className="max-w-4xl mx-auto space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Task Type
                </label>
                <select
                  value={taskType}
                  onChange={(e) => setTaskType(e.target.value as TaskType)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md bg-white text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                >
                  <option value="default">Default - General purpose</option>
                  <option value="document_analysis">Document Analysis - Analyze documents</option>
                  <option value="research">Research - Multi-source research</option>
                  <option value="calculation">Calculation - Mathematical tasks</option>
                  <option value="general">General - All tools available</option>
                </select>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="evaluateContext"
                  checked={evaluateContext}
                  onChange={(e) => setEvaluateContext(e.target.checked)}
                  className="w-4 h-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                />
                <label htmlFor="evaluateContext" className="text-sm text-gray-700 flex items-center gap-2">
                  <BarChart3 size={16} />
                  Evaluate context quality
                </label>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="useStreaming"
                  checked={useStreaming}
                  onChange={(e) => setUseStreaming(e.target.checked)}
                  className="w-4 h-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                />
                <label htmlFor="useStreaming" className="text-sm text-gray-700 flex items-center gap-2">
                  <Zap size={16} />
                  Enable streaming (shows reasoning in real-time)
                </label>
              </div>
            </div>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {messages.map((msg, index) => (
            <div 
              key={index} 
              className={`flex items-start gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
            >
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                msg.role === 'user' ? 'bg-gray-800 text-white' : 'bg-indigo-100 text-indigo-600'
              }`}>
                {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
              </div>
              
              <div className="flex flex-col gap-2 max-w-[80%]">
                {/* Reasoning Steps - Gray blocks that persist, shown BEFORE answer */}
                {msg.reasoning_steps && msg.reasoning_steps.length > 0 && (
                  <div className="space-y-2">
                    {msg.reasoning_steps.map((reasoning, idx) => (
                      <div key={idx} className="bg-gray-100 rounded-lg p-3 border border-gray-200">
                        <div className="flex items-start gap-2">
                          <Brain size={14} className="text-gray-600 mt-0.5 flex-shrink-0" />
                          <div className="flex-1">
                            <span className="text-xs font-medium text-gray-600">Thought {idx + 1}: </span>
                            <span className="text-xs text-gray-700 whitespace-pre-wrap">{reasoning}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Action Steps - Gray blocks that persist, shown BEFORE answer */}
                {msg.action_steps && msg.action_steps.length > 0 && (
                  <div className="space-y-2">
                    {msg.action_steps.map((action, idx) => (
                      <div key={idx} className="bg-gray-100 rounded-lg p-3 border border-gray-200">
                        <div className="flex items-start gap-2">
                          <Zap size={14} className="text-gray-600 mt-0.5 flex-shrink-0" />
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-xs font-medium text-gray-600">Action {idx + 1}:</span>
                              <span className="text-xs font-mono bg-gray-200 px-2 py-0.5 rounded text-gray-700">
                                {action.tool_name}
                              </span>
                            </div>
                            {Object.keys(action.input || {}).length > 0 && (
                              <div className="mb-1">
                                <span className="text-xs font-semibold text-gray-700">Input: </span>
                                <pre className="text-xs text-gray-600 mt-1 bg-white p-2 rounded border border-gray-300 overflow-x-auto">
                                  {JSON.stringify(action.input, null, 2)}
                                </pre>
                              </div>
                            )}
                            {action.output && (
                              <div>
                                <span className="text-xs font-semibold text-gray-700">Output: </span>
                                <p className="text-xs text-gray-600 mt-1 whitespace-pre-wrap bg-white p-2 rounded border border-gray-300 max-h-32 overflow-y-auto">
                                  {action.output.length > 500 ? action.output.substring(0, 500) + '...' : action.output}
                                </p>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Answer Content - White block, shown AFTER reasoning and actions */}
                {msg.content && (
                  <div className={`rounded-2xl px-4 py-3 shadow-sm ${
                msg.role === 'user' 
                  ? 'bg-blue-600 text-white rounded-tr-none' 
                  : 'bg-white border border-gray-100 text-gray-800 rounded-tl-none'
              }`}>
                 <p className="whitespace-pre-wrap leading-relaxed text-sm">
                   {msg.content}
                 </p>
                  </div>
                )}

                {/* Evaluation - shown after answer if available */}
                 {msg.evaluation && (
                  <div className="mt-2 bg-white rounded-2xl px-4 py-3 shadow-sm border border-gray-100">
                   <div className="space-y-3">
                     <div className="flex items-center gap-2 mb-2">
                       <BarChart3 size={16} className="text-indigo-600" />
                       <span className="text-xs font-semibold text-gray-700">Context Quality Evaluation</span>
                     </div>
                     
                     {/* Overall Score */}
                     <div className="bg-indigo-50 rounded-lg p-3">
                       <div className="flex items-center justify-between mb-1">
                         <span className="text-xs font-medium text-gray-700">Overall Quality</span>
                         <span className="text-sm font-bold text-indigo-600">
                           {(msg.evaluation.overall_quality_score * 100).toFixed(0)}%
                         </span>
                       </div>
                       <div className="w-full bg-gray-200 rounded-full h-2">
                         <div 
                           className="bg-indigo-600 h-2 rounded-full transition-all"
                           style={{ width: `${msg.evaluation.overall_quality_score * 100}%` }}
                         />
                       </div>
                       <p className="text-xs text-gray-600 mt-1">{msg.evaluation.recommendation}</p>
                     </div>

                     {/* Relevance Score */}
                     <div className="bg-green-50 rounded-lg p-3">
                       <div className="flex items-center justify-between mb-1">
                         <span className="text-xs font-medium text-gray-700">Relevance</span>
                         <span className="text-sm font-bold text-green-600">
                           {(msg.evaluation.relevance.relevance_score * 100).toFixed(0)}%
                         </span>
                       </div>
                       <div className="w-full bg-gray-200 rounded-full h-2">
                         <div 
                           className="bg-green-600 h-2 rounded-full transition-all"
                           style={{ width: `${msg.evaluation.relevance.relevance_score * 100}%` }}
                         />
                       </div>
                       <p className="text-xs text-gray-600 mt-1">{msg.evaluation.relevance.explanation}</p>
                       {msg.evaluation.relevance.key_points.length > 0 && (
                         <div className="mt-2">
                           <p className="text-xs font-medium text-gray-700 mb-1">Key Points:</p>
                           <ul className="text-xs text-gray-600 list-disc list-inside space-y-0.5">
                             {msg.evaluation.relevance.key_points.slice(0, 3).map((point, idx) => (
                               <li key={idx}>{point}</li>
                             ))}
                           </ul>
                         </div>
                       )}
                     </div>

                     {/* Completeness Score */}
                     <div className="bg-blue-50 rounded-lg p-3">
                       <div className="flex items-center justify-between mb-1">
                         <span className="text-xs font-medium text-gray-700">Completeness</span>
                         <span className="text-sm font-bold text-blue-600">
                           {(msg.evaluation.completeness.completeness_score * 100).toFixed(0)}%
                         </span>
                       </div>
                       <div className="w-full bg-gray-200 rounded-full h-2">
                         <div 
                           className="bg-blue-600 h-2 rounded-full transition-all"
                           style={{ width: `${msg.evaluation.completeness.completeness_score * 100}%` }}
                         />
                       </div>
                       <p className="text-xs text-gray-600 mt-1">{msg.evaluation.completeness.explanation}</p>
                       {msg.evaluation.completeness.missing_aspects.length > 0 && (
                         <div className="mt-2">
                           <p className="text-xs font-medium text-gray-700 mb-1">Missing Aspects:</p>
                           <ul className="text-xs text-gray-600 list-disc list-inside space-y-0.5">
                             {msg.evaluation.completeness.missing_aspects.slice(0, 3).map((aspect, idx) => (
                               <li key={idx}>{aspect}</li>
                             ))}
                           </ul>
                         </div>
                       )}
                     </div>
                     </div>
                   </div>
                 )}
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex items-center gap-2 text-gray-400 text-sm ml-12">
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-4 bg-white border-t border-gray-200">
          <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto relative flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your files..."
              className="flex-1 bg-gray-100 border-0 focus:ring-2 focus:ring-indigo-500 rounded-xl px-4 py-3 text-gray-800 placeholder-gray-400 outline-none transition-all"
            />
            <button 
              type="submit"
              disabled={!input.trim() || isLoading}
              className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white p-3 rounded-xl transition-colors"
            >
              <Send size={20} />
            </button>
          </form>
          <div className="text-center mt-2 flex items-center justify-center gap-4">
             <p className="text-xs text-gray-400">Agent remembers context within this session.</p>
             {taskType !== 'default' && (
               <span className="text-xs text-indigo-600 font-medium">Mode: {taskType.replace('_', ' ')}</span>
             )}
             {evaluateContext && (
               <span className="text-xs text-green-600 font-medium flex items-center gap-1">
                 <BarChart3 size={12} />
                 Evaluation ON
               </span>
             )}
          </div>
        </div>
      </div>
    </div>
  );
}