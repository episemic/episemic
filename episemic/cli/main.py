"""Main CLI application for Episemic Core."""

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..consolidation import ConsolidationEngine
from ..cortex import Cortex
from ..hippocampus import Hippocampus
from ..models import Memory, SearchQuery
from ..retrieval import RetrievalEngine

app = typer.Typer(
    name="episemic",
    help="🧠 Episemic Core - A brain-inspired memory system for AI agents",
    rich_markup_mode="rich",
)

console = Console()

# Global instances (in practice, these would be configured via settings)
hippocampus = None
cortex = None
consolidation_engine = None
retrieval_engine = None


def get_memory_system():
    global hippocampus, cortex, consolidation_engine, retrieval_engine

    if not all([hippocampus, cortex, consolidation_engine, retrieval_engine]):
        hippocampus = Hippocampus()
        cortex = Cortex()
        consolidation_engine = ConsolidationEngine(hippocampus, cortex)
        retrieval_engine = RetrievalEngine(hippocampus, cortex)

    return hippocampus, cortex, consolidation_engine, retrieval_engine


@app.command()
def init(
    qdrant_host: str = typer.Option("localhost", help="Qdrant host"),
    qdrant_port: int = typer.Option(6333, help="Qdrant port"),
    postgres_host: str = typer.Option("localhost", help="PostgreSQL host"),
    postgres_port: int = typer.Option(5432, help="PostgreSQL port"),
    postgres_db: str = typer.Option("episemic", help="PostgreSQL database name"),
):
    """Initialize the Episemic memory system."""
    try:
        console.print("🧠 Initializing Episemic Core...", style="bold blue")

        # Initialize components
        global hippocampus, cortex, consolidation_engine, retrieval_engine
        hippocampus = Hippocampus(qdrant_host, qdrant_port)
        cortex = Cortex(postgres_host, postgres_port, postgres_db)
        consolidation_engine = ConsolidationEngine(hippocampus, cortex)
        retrieval_engine = RetrievalEngine(hippocampus, cortex)

        console.print("✅ Episemic Core initialized successfully!", style="bold green")

    except Exception as e:
        console.print(f"❌ Failed to initialize: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def store(
    text: str = typer.Argument(..., help="Text content to store"),
    title: str = typer.Option("", help="Memory title"),
    source: str = typer.Option("cli", help="Memory source"),
    tags: list[str] = typer.Option([], help="Tags for the memory"),
):
    """Store a new memory in the system."""
    try:
        _, cortex, _, _ = get_memory_system()

        memory = Memory(
            title=title or text[:50] + "..." if len(text) > 50 else text,
            text=text,
            summary=text[:200] + "..." if len(text) > 200 else text,
            source=source,
            tags=tags,
        )

        async def store_memory():
            success = await cortex.store_memory(memory)
            return memory.id if success else None

        memory_id = asyncio.run(store_memory())

        if memory_id:
            console.print(f"✅ Memory stored with ID: [bold green]{memory_id}[/bold green]")
        else:
            console.print("❌ Failed to store memory", style="bold red")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"❌ Error storing memory: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, help="Number of results to return"),
    tags: list[str] = typer.Option([], help="Filter by tags"),
):
    """Search for memories."""
    try:
        _, _, _, retrieval_engine = get_memory_system()

        search_query = SearchQuery(
            query=query,
            top_k=top_k,
            filters={"tags": tags} if tags else {}
        )

        async def search_memories():
            return await retrieval_engine.search(search_query)

        results = asyncio.run(search_memories())

        if results:
            table = Table(title=f"Search Results for: {query}")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="magenta")
            table.add_column("Score", style="green")
            table.add_column("Tags", style="yellow")

            for result in results:
                table.add_row(
                    result.memory.id[:8] + "...",
                    result.memory.title[:50] + "..." if len(result.memory.title) > 50 else result.memory.title,
                    f"{result.score:.3f}",
                    ", ".join(result.memory.tags[:3])
                )

            console.print(table)
        else:
            console.print("No memories found matching your query.", style="yellow")

    except Exception as e:
        console.print(f"❌ Error searching memories: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def get(memory_id: str = typer.Argument(..., help="Memory ID to retrieve")):
    """Retrieve a specific memory by ID."""
    try:
        _, _, _, retrieval_engine = get_memory_system()

        async def get_memory():
            return await retrieval_engine.retrieve_by_id(memory_id)

        memory = asyncio.run(get_memory())

        if memory:
            panel = Panel(
                f"[bold]Title:[/bold] {memory.title}\n\n"
                f"[bold]Text:[/bold] {memory.text}\n\n"
                f"[bold]Source:[/bold] {memory.source}\n"
                f"[bold]Tags:[/bold] {', '.join(memory.tags)}\n"
                f"[bold]Created:[/bold] {memory.created_at}\n"
                f"[bold]Access Count:[/bold] {memory.access_count}",
                title=f"Memory: {memory_id[:8]}...",
                border_style="blue"
            )
            console.print(panel)
        else:
            console.print(f"❌ Memory {memory_id} not found", style="bold red")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"❌ Error retrieving memory: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def consolidate(
    memory_id: str | None = typer.Option(None, help="Specific memory ID to consolidate"),
    auto: bool = typer.Option(False, help="Run auto-consolidation sweep"),
):
    """Consolidate memories from hippocampus to cortex."""
    try:
        _, _, consolidation_engine, _ = get_memory_system()

        if auto:
            async def run_auto_consolidation():
                return await consolidation_engine.auto_consolidation_sweep()

            count = asyncio.run(run_auto_consolidation())
            console.print(f"✅ Auto-consolidation completed. {count} memories processed.", style="bold green")

        elif memory_id:
            async def consolidate_single():
                return await consolidation_engine.consolidate_memory(memory_id)

            success = asyncio.run(consolidate_single())
            if success:
                console.print(f"✅ Memory {memory_id} consolidated successfully", style="bold green")
            else:
                console.print(f"❌ Failed to consolidate memory {memory_id}", style="bold red")
                raise typer.Exit(1)
        else:
            console.print("❌ Please specify either --memory-id or --auto", style="bold red")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"❌ Error during consolidation: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def health():
    """Check the health status of all system components."""
    try:
        hippocampus, cortex, consolidation_engine, retrieval_engine = get_memory_system()

        table = Table(title="🏥 System Health Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="magenta")

        # Check hippocampus
        hippo_health = hippocampus.health_check()
        hippo_status = "✅ Healthy" if all(hippo_health.values()) else "❌ Unhealthy"
        table.add_row("Hippocampus", hippo_status)

        # Check cortex
        cortex_healthy = cortex.health_check()
        cortex_status = "✅ Healthy" if cortex_healthy else "❌ Unhealthy"
        table.add_row("Cortex", cortex_status)

        # Check consolidation engine
        consolidation_health = consolidation_engine.health_check()
        consolidation_status = "✅ Healthy" if all(v for v in consolidation_health.values() if isinstance(v, (bool, dict))) else "❌ Unhealthy"
        table.add_row("Consolidation Engine", consolidation_status)

        # Check retrieval engine
        retrieval_health = retrieval_engine.health_check()
        retrieval_status = "✅ Healthy" if all(v for v in retrieval_health.values() if isinstance(v, (bool, dict))) else "❌ Unhealthy"
        table.add_row("Retrieval Engine", retrieval_status)

        console.print(table)

    except Exception as e:
        console.print(f"❌ Error checking health: {e}", style="bold red")
        raise typer.Exit(1) from e


@app.command()
def version():
    """Show version information."""
    from .. import __version__
    console.print(f"🧠 Episemic Core v{__version__}", style="bold blue")


if __name__ == "__main__":
    app()
