"""Sightline CLI — Command-line interface for the analysis platform.

Commands:
  sightline process <vod>     Process a single VOD
  sightline batch <dir>       Batch process all VODs in directory
  sightline download <url>    Download VODs from YouTube playlist
  sightline scout <team>      Generate scouting report for an opponent
  sightline serve             Start the web platform
  sightline label             Launch labeling tool for training data
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="sightline",
    help="AI-powered CoD League VOD analysis platform",
    no_args_is_help=True,
)
console = Console()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def process(
    vod_path: str = typer.Argument(help="Path to VOD file"),
    opponent: str = typer.Option("", help="Opponent team name"),
    event: str = typer.Option("", help="Event/tournament name"),
    config: str = typer.Option("config.yaml", help="Config file path"),
):
    """Process a single VOD through the full pipeline."""
    from cod_analyst.config import load_config
    from cod_analyst.db.schemas import init_db
    from cod_analyst.pipeline import SightlinePipeline

    cfg = load_config(config)
    init_db(cfg.database.url)

    pipeline = SightlinePipeline(cfg)
    
    console.print(f"[bold green]Starting Pipeline[/bold green] on {vod_path}")
    console.print(f"Opponent: {opponent or 'Unknown'} | Event: {event or 'Unknown'}")
    
    # Run the full pipeline (it has its own logging)
    frames_processed = pipeline.process_vod(
        vod_path=vod_path,
        opponent=opponent,
        event_name=event,
    )

    console.print(f"\n[green]✓[/green] Processing complete: {frames_processed} frames processed")


@app.command()
def batch(
    vod_dir: str = typer.Argument(help="Directory containing VOD files"),
    config: str = typer.Option("config.yaml", help="Config file path"),
):
    """Batch process all VODs in a directory."""
    vod_path = Path(vod_dir)
    if not vod_path.is_dir():
        console.print(f"[red]Error:[/red] {vod_dir} is not a directory")
        raise typer.Exit(1)

    vods = sorted(vod_path.glob("*.mp4")) + sorted(vod_path.glob("*.mkv"))
    console.print(f"Found {len(vods)} VOD files")

    for i, vod in enumerate(vods, 1):
        console.print(f"\n[bold][{i}/{len(vods)}] Processing: {vod.name}[/bold]")
        process(str(vod), config=config)


@app.command()
def download(
    playlist_url: str = typer.Argument(help="YouTube playlist URL"),
    keyword: str = typer.Option("faze", help="Filter keyword for video titles"),
    limit: int = typer.Option(20, help="Maximum number of videos to download"),
    output_dir: str = typer.Option("", help="Output directory (default: config vods_dir)"),
    config: str = typer.Option("config.yaml", help="Config file path"),
):
    """Download VODs from a YouTube playlist."""
    from cod_analyst.config import load_config
    from cod_analyst.ingest.downloader import list_playlist, download_vods

    cfg = load_config(config)
    out = output_dir or cfg.paths.vods_dir

    console.print(f"[bold]Fetching playlist:[/bold] {playlist_url}")
    console.print(f"[bold]Filter:[/bold] '{keyword}' | [bold]Limit:[/bold] {limit}")

    entries = list_playlist(playlist_url, keyword_filter=keyword, limit=limit)

    if not entries:
        console.print("[yellow]No matching videos found[/yellow]")
        raise typer.Exit()

    # Display matches
    table = Table(title=f"Found {len(entries)} matching videos")
    table.add_column("#", style="dim")
    table.add_column("Title")
    table.add_column("Opponent")
    table.add_column("Duration")

    for i, entry in enumerate(entries, 1):
        dur = f"{entry.duration_sec / 60:.0f}m" if entry.duration_sec else "?"
        table.add_row(str(i), entry.title, entry.opponent or "?", dur)

    console.print(table)

    if typer.confirm(f"Download {len(entries)} videos to {out}?"):
        downloaded = download_vods(entries, out, cfg.downloader.default_resolution)
        console.print(f"\n[green]✓[/green] Downloaded {len(downloaded)}/{len(entries)} videos")


@app.command()
def scout(
    team_name: str = typer.Argument(help="Opponent team name to scout"),
    map_name: str = typer.Option("", help="Filter by map"),
    config: str = typer.Option("config.yaml", help="Config file path"),
):
    """Generate a scouting report for an opponent team."""
    from cod_analyst.config import load_config
    from cod_analyst.db.schemas import get_session, DBPlaystyleProfile
    from sqlmodel import select

    cfg = load_config(config)
    session = get_session(cfg.database.url)

    profiles = session.exec(
        select(DBPlaystyleProfile).where(DBPlaystyleProfile.team_name == team_name)
    ).all()

    if not profiles:
        console.print(f"[yellow]No data found for team '{team_name}'[/yellow]")
        console.print("Process some VODs first with: sightline process <vod> --opponent <team>")
        raise typer.Exit()

    table = Table(title=f"Scouting: {team_name}")
    table.add_column("Map")
    table.add_column("Mode")
    table.add_column("Pace")
    table.add_column("Trade Rate")
    table.add_column("First Blood")

    for p in profiles:
        table.add_row(
            p.map_name, p.mode,
            f"{p.pace_score:.2f}",
            f"{p.trade_rate:.0%}",
            f"{p.first_blood_rate:.0%}",
        )

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
    config: str = typer.Option("config.yaml", help="Config file path"),
):
    """Start the Sightline web platform."""
    import uvicorn

    console.print(f"[bold green]Starting Sightline[/bold green] at http://{host}:{port}")
    uvicorn.run(
        "cod_analyst.api.routes:app",
        host=host,
        port=port,
        reload=True,
    )


@app.command()
def label(
    data_type: str = typer.Argument(help="Type of data to label: 'mode', 'minimap', 'weapon'"),
    input_dir: str = typer.Option("", help="Directory of frames to label"),
):
    """Launch labeling tool for creating training data."""
    console.print(f"[bold]Labeling tool:[/bold] {data_type}")
    console.print("This feature will launch an interactive labeling UI.")
    console.print("[yellow]Coming soon — requires frame samples first.[/yellow]")


@app.command()
def init(
    config: str = typer.Option("config.yaml", help="Config file path"),
):
    """Initialize database and directory structure."""
    from cod_analyst.config import load_config
    from cod_analyst.db.schemas import init_db

    cfg = load_config(config)
    init_db(cfg.database.url)
    console.print("[green]✓[/green] Database initialized")
    console.print("[green]✓[/green] Directory structure created")


if __name__ == "__main__":
    app()
