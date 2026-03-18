ADAS Simulation Project (Agent Instructions)
Business Requirements

    Ein funktionsfähiges MVP einer 2D-Autobahnsimulation in Python.

    Fokus: ADAS (Advanced Driver Assistance Systems), spezifisch Spurhalteassistent (LKA) und Abstandsregeltempomat (ACC).

    Das System muss Sensordaten (Abstand zum Vorderkasten, Linienposition) simulieren.

    UI: Ein Fenster, das die Straße, das eigene Fahrzeug und andere Verkehrsteilnehmer flüssig darstellt.

    Interaktivität: Der Nutzer kann Hindernisse per Mausklick platzieren oder die Zielgeschwindigkeit ändern.

    Keine Persistenz nötig; Fokus liegt auf der Echtzeit-Logik und Visualisierung.

Technical Details

    Implementierung in Python 3.10+.

    Hauptbibliothek für Visualisierung: Pygame (wegen der einfachen Frame-basierten Steuerung).

    Mathematik: Numpy für Vektorberechnungen der Fahrzeugdynamik.

    Architektur: Striktes MVC-Modell (Model: Fahrzeugphysik, View: Pygame-Renderer, Controller: ADAS-Logik).

    Projektstruktur: Alle Quelldateien im Ordner src/.

Color Scheme & Visuals

    Hintergrund (Asphalt): #2c3e50 (Dark Gray/Blue)

    Markierungen: #ffffff (Weiß)

    Ego-Fahrzeug: #209dd7 (Blue Primary)

    Sensoren/Radar-Kegel: #ecad0a (Accent Yellow) mit Transparenz

    Warnzustände: #e74c3c (Rot), wenn der Abstand zu gering ist.

    UI-Text: #888888 (Gray) für Telemetrie-Daten am Rand.

Strategy

    Scaffolding: Erstellen der Verzeichnisstruktur, .gitignore und requirements.txt.

    Physics Engine: Implementierung eines "Bicycle Models" für das Fahrzeug und einfacher PID-Regler für ACC/LKA.

    Sensor Simulation: Logik für "Raycasting", um Abstände zu Objekten vor dem Fahrzeug zu messen.

    Visualisierungs-Layer: Aufbau des Pygame-Screens mit flüssigen 60 FPS.

    Integration & Testing: Validierung der Regelschleifen (schwingt das Auto? bremst es rechtzeitig?).

Coding Standards

    Idiomatisches Python: Nutze Type-Hints (typing) für alle Funktionen.

    Modularität: Keine "Giga-Files". Trenne die ADAS-Logik strikt von der Pygame-Rendering-Schleife.

    Simplicity: Keine komplexe 3D-Engine. 2D-Top-Down reicht völlig aus, solange die Logik "anspruchsvoll" ist.

    Dokumentation: Docstrings für Klassen und komplexe mathematische Formeln. Keine Emojis in Kommentaren oder README.
