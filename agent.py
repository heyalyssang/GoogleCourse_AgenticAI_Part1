
import os
import sys
from pathlib import Path
import datetime

from dotenv import load_dotenv
from google.adk.agents import Agent, LoopAgent
from google.adk.tools import agent_tool

# --- env/config ---------------------------
load_dotenv()

# MODEL = os.getenv("MODEL", "gemini-flash-latest")
# gemini model currently experiencing high demand, cannot run the latest model...
MODEL = os.getenv("MODEL", "gemini-flash-lite-latest")


# --- Sub-Agent: Planner -------------------
blog_planner = Agent(
    name="BlogPlanner",
    model=MODEL,
    description="Creates a practical, skimmable outline in Markdown.",
    instruction="""
        You are a technical content strategist. Produce a clear Markdown outline with:
        - Title
        - Short intro
        - 4-6 main sections (each with 2-3 bullets)
        - Conclusion

        If `codebase_context` exists in state, weave in specific sections/snippets.
        Return only the outline in Markdown.
    """,
    output_key="blog_outline"
)

# --- Outline Validation Checker Class -----
class OutlineValidationChecker(Agent):
    def __init__(self):
        super().__init__(
            name="OutlineValidationChecker",
            model=MODEL,
            description="Validates that the outline is usable.",
            instruction="""
                Check the outline in state `blog_outline`. If it has a title, intro, 4-6 sections, and a conclusion, respond exactly "ok".
                Otherwise respond exactly "retry" and list missing pieces.
            """,
            output_key="validation result",
        )

# --- The Robust Blog Planner Loop ---------
robust_blog_planner = LoopAgent(
    name="RobustBlogPlanner",
    description="Retries planning if validation fails.",
    sub_agents=[
        blog_planner,
        OutlineValidationChecker()],
    max_iterations=3,
)

# --- Sub-Agent: Writer --------------------
blog_writer = Agent(
    name="BlogWriter",
    model=MODEL,
    description="Writes a technical blog post from the outline.",
    instruction="""
        Write a complete Markdown article from the outline in `blog_outline`.
        
        Guidelines: 
        - Audience: Software engineers; skip basics and focus on practical insight. 
        - Explain both the 'how' and 'why'/
        - Include concise code snippets when helpful.
        - Follow the outline's structure (H2/H3).
        - Output only the final article in Markdown (no fence around the whole post).
    """,
    output_key="blog_post",
)

# --- Blog Post Validation Checker Class ---
class BlogPostValidationChecker(Agent):
    def __init__(self):
        super().__init__(
            name="BlogPostValidationChecker",
            model=MODEL,
            description="Validates the final post.",
            instruction="""
                Check `blog_post` for: intro, clear sections matching the outline, conclusion, and technical clarity.
                If passes, respond "ok". Else respond "retry" with the specific fixes.
            """,
            output_key="validation_result",
        )

# --- The Robust Blog Writer Loop ----------
robust_blog_writer = LoopAgent(
    name="RobustBlogWriter",
    description="Retries writing if validation fails.",
    sub_agents=[
        blog_writer,
        BlogPostValidationChecker()],
    max_iterations=3,
)

# --- Wrap Each Loop as Tools --------------
# Expose planner/writer as tools so the root agent can call them explicitly
planner_tool = agent_tool.AgentTool(agent=robust_blog_planner)
writer_tool  = agent_tool.AgentTool(agent=robust_blog_writer)


# --- Root Agent: Trends → Plan → Write ----
root_agent = Agent(
    name="Blogger",
    model=MODEL,
    description="Minimal multi-agent blogger that plans and writes.",
    instruction=f"""
        If the user gives a topic:
        1) Call the planner tool to generate the outline.
        2) Call the writer tool to produce the full draft.
        3) End with 3 alternate titles and 2 tweet-length hook.
        
        If the trends tool fails or times out, continue with a sensible outline and draft anyway.

        Date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    """,
    tools=[
        planner_tool,
        writer_tool,
    ],
)