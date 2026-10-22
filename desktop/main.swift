import Cocoa
import Foundation

// MARK: - Configuration

struct AppConfig: Codable {
    var repoPath: String
    var pythonPath: String
    var port: Int
    var autoStart: Bool
    var stopOnQuit: Bool

    static var supportDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/MLX Controller", isDirectory: true)
    }

    static var configURL: URL {
        supportDirectory.appendingPathComponent("config.json")
    }

    static var logURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/MLX Controller/server.log")
    }

    static func load() -> AppConfig {
        var config = AppConfig(repoPath: "", pythonPath: "", port: 9998, autoStart: true, stopOnQuit: true)

        if let data = try? Data(contentsOf: configURL),
           let saved = try? JSONDecoder().decode(AppConfig.self, from: data) {
            config = saved
        }

        let info = Bundle.main.infoDictionary ?? [:]
        if config.repoPath.isEmpty, let bundled = info["MLXRepoPath"] as? String, !bundled.isEmpty {
            config.repoPath = bundled
        }
        if config.pythonPath.isEmpty, let bundled = info["MLXPythonPath"] as? String, !bundled.isEmpty {
            config.pythonPath = bundled
        }
        if config.repoPath.isEmpty {
            let installed = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("mac-mlx-control-center", isDirectory: true)
                .path
            if FileManager.default.fileExists(atPath: installed + "/mlx_gui_server.py") {
                config.repoPath = installed
            }
        }
        if config.pythonPath.isEmpty {
            config.pythonPath = detectPython()
        }
        return config
    }

    static func detectPython() -> String {
        let candidates = ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"]
        for candidate in candidates where FileManager.default.isExecutableFile(atPath: candidate) {
            return candidate
        }
        return "/usr/bin/python3"
    }

    func save() {
        do {
            try FileManager.default.createDirectory(at: AppConfig.supportDirectory, withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(self)
            try data.write(to: AppConfig.configURL, options: .atomic)
        } catch {
            NSLog("MLX Controller: could not save config: \(error)")
        }
    }

    var serverScript: String { repoPath + "/mlx_gui_server.py" }

    func validationError() -> String? {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: repoPath, isDirectory: &isDirectory), isDirectory.boolValue else {
            return "Project folder not found at \(repoPath)."
        }
        guard FileManager.default.fileExists(atPath: serverScript) else {
            return "mlx_gui_server.py was not found in \(repoPath)."
        }
        guard FileManager.default.isExecutableFile(atPath: pythonPath) else {
            return "Python 3 was not found at \(pythonPath)."
        }
        return nil
    }

    var dashboardURL: URL? { URL(string: "http://127.0.0.1:\(port)") }
}

// MARK: - Server lifecycle

final class ServerController: NSObject {
    enum State {
        case stopped
        case starting
        case running
        case stopping
        case error(String)

        var label: String {
            switch self {
            case .stopped: return "Stopped"
            case .starting: return "Starting…"
            case .running: return "Running"
            case .stopping: return "Stopping…"
            case .error: return "Couldn’t start"
            }
        }

        var isBusy: Bool {
            switch self {
            case .starting, .stopping: return true
            default: return false
            }
        }

        var isRunning: Bool {
            if case .running = self { return true }
            return false
        }
    }

    struct PortOwner {
        let pid: pid_t
        let command: String
    }

    private(set) var state: State = .stopped {
        didSet { onStateChange?(state) }
    }

    var config: AppConfig
    var onStateChange: ((State) -> Void)?
    var onLog: ((String) -> Void)?

    var isQuitting = false

    private var process: Process?
    private var logHandle: FileHandle?
    private var pipe: Pipe?
    private var intentionalStop = false
    private var external = false

    init(config: AppConfig) {
        self.config = config
        super.init()
    }

    var ownsRunningProcess: Bool { process?.isRunning == true }

    var statusDetail: String {
        switch state {
        case .stopped:
            return "Port \(config.port) · not running"
        case .starting:
            return "Launching mlx_gui_server.py…"
        case .running:
            return external ? "Port \(config.port) · started outside this app" : "Port \(config.port) · dashboard ready"
        case .stopping:
            return "Waiting for the server to exit…"
        case .error(let message):
            return message
        }
    }

    func updateRepo(_ path: String) {
        config.repoPath = path
        config.save()
    }

    // MARK: Health and port helpers

    func healthCheck(completion: @escaping (Bool) -> Void) {
        guard let url = URL(string: "http://127.0.0.1:\(config.port)/api/status") else {
            completion(false)
            return
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 1.0
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let task = URLSession.shared.dataTask(with: request) { _, response, _ in
            let ok = (response as? HTTPURLResponse)?.statusCode == 200
            DispatchQueue.main.async { completion(ok) }
        }
        task.resume()
    }

    private func runCommand(_ path: String, _ arguments: [String]) -> String {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: path)
        task.arguments = arguments
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        do {
            try task.run()
            task.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8) ?? ""
        } catch {
            return ""
        }
    }

    func portOwner() -> PortOwner? {
        let output = runCommand("/usr/sbin/lsof", ["-nP", "-iTCP:\(config.port)", "-sTCP:LISTEN", "-Fpc"])
        var pid: pid_t?
        var command = "unknown"
        for line in output.split(separator: "\n") {
            if line.hasPrefix("p"), let value = pid_t(line.dropFirst()) {
                pid = value
            } else if line.hasPrefix("c") {
                command = String(line.dropFirst())
            }
        }
        guard let found = pid else { return nil }
        let full = runCommand("/bin/ps", ["-p", "\(found)", "-o", "command="]).trimmingCharacters(in: .whitespacesAndNewlines)
        if !full.isEmpty { command = full }
        return PortOwner(pid: found, command: command)
    }

    private func isControlCenterProcess(_ pid: pid_t) -> Bool {
        let full = runCommand("/bin/ps", ["-p", "\(pid)", "-o", "command="])
        return full.contains("mlx_gui_server.py")
    }

    private func shortProcessName(_ command: String) -> String {
        guard let first = command.split(separator: " ").first else { return command }
        let name = URL(fileURLWithPath: String(first)).lastPathComponent
        return name.isEmpty ? String(first) : name
    }

    // MARK: Start / stop

    func refresh(completion: (() -> Void)? = nil) {
        healthCheck { [weak self] healthy in
            guard let self else { return }
            if healthy {
                self.external = (self.process == nil)
                self.state = .running
            } else if let owner = self.portOwner() {
                if self.process != nil {
                    self.state = .starting
                } else {
                    self.external = false
                    self.state = .error("Port \(self.config.port) is in use by \(self.shortProcessName(owner.command)) (PID \(owner.pid)). Quit that app or free the port.")
                }
            } else if self.process?.isRunning == true {
                self.state = .starting
            } else {
                self.external = false
                self.state = .stopped
            }
            completion?()
        }
    }

    func start(openDashboard: Bool = false) {
        guard !state.isBusy else { return }
        if state.isRunning {
            if openDashboard { openDashboardURL() }
            return
        }
        if let problem = config.validationError() {
            state = .error(problem)
            return
        }

        state = .starting
        healthCheck { [weak self] healthy in
            guard let self else { return }
            if healthy {
                self.external = (self.process == nil)
                self.state = .running
                if openDashboard { self.openDashboardURL() }
                return
            }
            if let owner = self.portOwner() {
                if self.isControlCenterProcess(owner.pid) {
                    // Another launcher owns the control center. Adopt it once healthy.
                    self.external = true
                    self.pollUntilReady(deadline: Date().addingTimeInterval(15)) { ok in
                        if ok {
                            self.state = .running
                            if openDashboard { self.openDashboardURL() }
                        } else {
                            self.state = .error("Port \(self.config.port) is busy with a control center that is not responding.")
                        }
                    }
                } else {
                    self.state = .error("Port \(self.config.port) is in use by \(self.shortProcessName(owner.command)) (PID \(owner.pid)). Quit that app or free the port, then try again.")
                }
                return
            }
            self.spawn(openAfterStart: openDashboard)
        }
    }

    private func spawn(openAfterStart: Bool = false) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: config.pythonPath)
        process.arguments = ["-u", config.serverScript]
        process.currentDirectoryURL = URL(fileURLWithPath: config.repoPath, isDirectory: true)

        var environment = ProcessInfo.processInfo.environment
        environment["PATH"] = "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        prepareLog()
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            try? self?.logHandle?.write(contentsOf: data)
            if let text = String(data: data, encoding: .utf8) {
                DispatchQueue.main.async { self?.ingest(text) }
            }
        }

        process.terminationHandler = { [weak self] finished in
            DispatchQueue.main.async {
                guard let self else { return }
                if self.process === finished {
                    self.process = nil
                }
                if self.intentionalStop {
                    self.intentionalStop = false
                    if case .stopping = self.state {
                        self.state = .stopped
                    }
                } else if !self.isQuitting {
                    self.state = .error("The server exited (status \(finished.terminationStatus)). See the log below.")
                }
            }
        }

        do {
            try process.run()
        } catch {
            state = .error("Could not launch Python: \(error.localizedDescription)")
            return
        }

        self.process = process
        self.pipe = pipe
        self.external = false
        self.intentionalStop = false
        self.state = .starting
        append("launching \(config.pythonPath) \(config.serverScript)")

        let deadline = Date().addingTimeInterval(35)
        pollUntilReady(deadline: deadline) { [weak self] ok in
            guard let self else { return }
            if ok {
                self.state = .running
                if openAfterStart { self.openDashboardURL() }
            } else if case .starting = self.state {
                self.state = .error("The server did not become ready within 35 seconds. Check the log below.")
            }
        }
    }

    private func pollUntilReady(deadline: Date, completion: @escaping (Bool) -> Void) {
        if Date() >= deadline {
            completion(false)
            return
        }
        if !external, let process, !process.isRunning {
            completion(false)
            return
        }
        healthCheck { [weak self] ok in
            guard let self else { return }
            if ok {
                completion(true)
            } else {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                    self.pollUntilReady(deadline: deadline, completion: completion)
                }
            }
        }
    }

    func stop(completion: (() -> Void)? = nil) {
        if let process, process.isRunning {
            intentionalStop = true
            state = .stopping
            process.terminate()
            waitForProcessExit(process, deadline: Date().addingTimeInterval(6)) { [weak self] exited in
                guard let self else { completion?(); return }
                if !exited {
                    Darwin.kill(process.processIdentifier, SIGKILL)
                }
                self.finishStopped()
                completion?()
            }
            return
        }

        if let owner = portOwner() {
            guard isControlCenterProcess(owner.pid) else {
                state = .error("Refusing to stop a process that is not the MLX control center.")
                completion?()
                return
            }
            state = .stopping
            Darwin.kill(owner.pid, SIGTERM)
            waitForPIDExit(owner.pid, deadline: Date().addingTimeInterval(6)) { [weak self] exited in
                guard let self else { completion?(); return }
                if !exited {
                    Darwin.kill(owner.pid, SIGKILL)
                }
                self.finishStopped()
                completion?()
            }
            return
        }

        finishStopped()
        completion?()
    }

    private func finishStopped() {
        pipe?.fileHandleForReading.readabilityHandler = nil
        try? logHandle?.close()
        logHandle = nil
        process = nil
        pipe = nil
        external = false
        intentionalStop = false
        state = .stopped
        append("server stopped")
    }

    private func waitForProcessExit(_ process: Process, deadline: Date, completion: @escaping (Bool) -> Void) {
        if !process.isRunning {
            completion(true)
            return
        }
        if Date() >= deadline {
            completion(false)
            return
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { [weak self] in
            self?.waitForProcessExit(process, deadline: deadline, completion: completion)
        }
    }

    private func waitForPIDExit(_ pid: pid_t, deadline: Date, completion: @escaping (Bool) -> Void) {
        if Darwin.kill(pid, 0) != 0 {
            completion(true)
            return
        }
        if Date() >= deadline {
            completion(false)
            return
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { [weak self] in
            self?.waitForPIDExit(pid, deadline: deadline, completion: completion)
        }
    }

    func restart(completion: (() -> Void)? = nil) {
        guard !state.isBusy else { return }
        stop { [weak self] in
            self?.start()
            completion?()
        }
    }

    func openDashboard() {
        if state.isRunning {
            openDashboardURL()
        } else {
            start(openDashboard: true)
        }
    }

    func openDashboardURL() {
        guard let url = config.dashboardURL else { return }
        NSWorkspace.shared.open(url)
    }

    // MARK: Logging

    private func prepareLog() {
        do {
            try FileManager.default.createDirectory(
                at: AppConfig.logURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            if !FileManager.default.fileExists(atPath: AppConfig.logURL.path) {
                FileManager.default.createFile(atPath: AppConfig.logURL.path, contents: nil)
            }
            let handle = try FileHandle(forWritingTo: AppConfig.logURL)
            try handle.seekToEnd()
            logHandle = handle
            let stamp = ISO8601DateFormatter().string(from: Date())
            try handle.write(contentsOf: Data("\n=== MLX Controller \(stamp) ===\n".utf8))
        } catch {
            NSLog("MLX Controller: could not open log: \(error)")
        }
    }

    private func append(_ line: String) {
        DispatchQueue.main.async { [weak self] in
            self?.onLog?(line + "\n")
        }
    }

    private func ingest(_ text: String) {
        onLog?(text)
    }
}

// MARK: - Main window

final class MainViewController: NSViewController {
    private let server: ServerController

    private let statusDot = NSView()
    private let statusLabel = NSTextField(labelWithString: "Stopped")
    private let detailLabel = NSTextField(labelWithString: "")
    private let spinner = NSProgressIndicator()
    private let startButton = NSButton(title: "Start", target: nil, action: nil)
    private let stopButton = NSButton(title: "Stop", target: nil, action: nil)
    private let restartButton = NSButton(title: "Restart", target: nil, action: nil)
    private let openButton = NSButton(title: "Open Dashboard", target: nil, action: nil)
    private let pathLabel = NSTextField(labelWithString: "")
    private let logView = NSTextView()
    private var logLines: [String] = []
    private var logBuffer = ""

    private let background = NSColor(calibratedRed: 0.043, green: 0.055, blue: 0.094, alpha: 1)
    private let cardColor = NSColor(white: 1, alpha: 0.055)
    private let accent = NSColor(calibratedRed: 0.37, green: 0.88, blue: 1.0, alpha: 1)
    private let primaryText = NSColor(calibratedWhite: 0.95, alpha: 1)
    private let secondaryText = NSColor(calibratedWhite: 0.62, alpha: 1)

    init(server: ServerController) {
        self.server = server
        super.init(nibName: nil, bundle: nil)
        server.onStateChange = { [weak self] state in self?.render(state) }
        server.onLog = { [weak self] text in self?.appendLog(text) }
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func loadView() {
        let root = NSView()
        root.wantsLayer = true
        root.layer?.backgroundColor = background.cgColor
        view = root
        preferredContentSize = NSSize(width: 460, height: 460)
        buildInterface()
        render(server.state)
        logView.string = "No server output yet."
        pathLabel.stringValue = abbreviatedPath(server.config.repoPath)
        pathLabel.toolTip = server.config.repoPath
    }

    private func label(_ text: String, size: CGFloat, weight: NSFont.Weight = .regular, color: NSColor? = nil) -> NSTextField {
        let field = NSTextField(labelWithString: text)
        field.font = NSFont.systemFont(ofSize: size, weight: weight)
        field.textColor = color ?? primaryText
        field.lineBreakMode = .byTruncatingTail
        return field
    }

    private func styleButton(_ button: NSButton, accent: Bool = false) {
        button.bezelStyle = .rounded
        button.controlSize = .large
        button.font = NSFont.systemFont(ofSize: 13, weight: .semibold)
        button.focusRingType = .none
        button.target = self
        applyButtonStyle(button, enabled: true, accent: accent)
    }

    private func applyButtonStyle(_ button: NSButton, enabled: Bool, accent: Bool) {
        let titleColor: NSColor
        if !enabled {
            titleColor = NSColor(white: 0.78, alpha: 1)
            button.bezelColor = NSColor(white: 1, alpha: 0.075)
        } else if accent {
            titleColor = NSColor.black
            button.bezelColor = self.accent
        } else {
            titleColor = primaryText
            button.bezelColor = NSColor(white: 1, alpha: 0.14)
        }
        button.attributedTitle = NSAttributedString(
            string: button.title,
            attributes: [
                .foregroundColor: titleColor,
                .font: NSFont.systemFont(ofSize: 13, weight: .semibold),
            ]
        )
    }

    private func buildInterface() {
        let icon = NSImageView()
        icon.imageScaling = .scaleProportionallyUpOrDown
        if let path = Bundle.main.path(forResource: "AppIcon", ofType: "icns"), let image = NSImage(contentsOfFile: path) {
            icon.image = image
        } else {
            icon.image = NSApp.applicationIconImage
        }
        icon.translatesAutoresizingMaskIntoConstraints = false
        icon.widthAnchor.constraint(equalToConstant: 62).isActive = true
        icon.heightAnchor.constraint(equalToConstant: 62).isActive = true

        let title = label("LLM Controller", size: 23, weight: .bold)
        let subtitle = label("macOS LLM Control Center", size: 12, weight: .regular, color: secondaryText)
        let titleStack = NSStackView(views: [title, subtitle])
        titleStack.orientation = .vertical
        titleStack.alignment = .leading
        titleStack.spacing = 2

        let header = NSStackView(views: [icon, titleStack])
        header.orientation = .horizontal
        header.alignment = .centerY
        header.spacing = 14

        let card = NSView()
        card.wantsLayer = true
        card.layer?.backgroundColor = cardColor.cgColor
        card.layer?.cornerRadius = 13
        card.translatesAutoresizingMaskIntoConstraints = false
        card.heightAnchor.constraint(equalToConstant: 82).isActive = true

        statusDot.wantsLayer = true
        statusDot.layer?.cornerRadius = 5
        statusDot.translatesAutoresizingMaskIntoConstraints = false
        statusDot.widthAnchor.constraint(equalToConstant: 10).isActive = true
        statusDot.heightAnchor.constraint(equalToConstant: 10).isActive = true

        statusLabel.font = NSFont.systemFont(ofSize: 16, weight: .semibold)
        statusLabel.textColor = primaryText
        detailLabel.font = NSFont.systemFont(ofSize: 11.5, weight: .regular)
        detailLabel.textColor = secondaryText
        detailLabel.maximumNumberOfLines = 2
        detailLabel.lineBreakMode = .byWordWrapping
        detailLabel.preferredMaxLayoutWidth = 330

        spinner.style = .spinning
        spinner.controlSize = .small
        spinner.isDisplayedWhenStopped = false
        spinner.translatesAutoresizingMaskIntoConstraints = false

        let textStack = NSStackView(views: [statusLabel, detailLabel])
        textStack.orientation = .vertical
        textStack.alignment = .leading
        textStack.spacing = 3

        card.addSubview(statusDot)
        card.addSubview(textStack)
        card.addSubview(spinner)
        textStack.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            statusDot.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 18),
            statusDot.centerYAnchor.constraint(equalTo: card.centerYAnchor),
            textStack.leadingAnchor.constraint(equalTo: statusDot.trailingAnchor, constant: 12),
            textStack.centerYAnchor.constraint(equalTo: card.centerYAnchor),
            textStack.trailingAnchor.constraint(lessThanOrEqualTo: spinner.leadingAnchor, constant: -10),
            spinner.trailingAnchor.constraint(equalTo: card.trailingAnchor, constant: -18),
            spinner.centerYAnchor.constraint(equalTo: card.centerYAnchor),
        ])

        startButton.title = "Start"
        stopButton.title = "Stop"
        restartButton.title = "Restart"
        styleButton(startButton, accent: true)
        styleButton(stopButton)
        styleButton(restartButton)
        startButton.target = self
        stopButton.target = self
        restartButton.target = self
        startButton.action = #selector(startTapped)
        stopButton.action = #selector(stopTapped)
        restartButton.action = #selector(restartTapped)

        let buttonRow = NSStackView(views: [startButton, stopButton, restartButton])
        buttonRow.orientation = .horizontal
        buttonRow.distribution = .fillEqually
        buttonRow.spacing = 8
        buttonRow.translatesAutoresizingMaskIntoConstraints = false
        buttonRow.heightAnchor.constraint(equalToConstant: 36).isActive = true

        openButton.title = "Open Dashboard"
        styleButton(openButton)
        openButton.bezelColor = NSColor(white: 1, alpha: 0.09)
        openButton.target = self
        openButton.action = #selector(openTapped)
        openButton.translatesAutoresizingMaskIntoConstraints = false
        openButton.heightAnchor.constraint(equalToConstant: 40).isActive = true

        let logTitle = label("SERVER LOG", size: 9.5, weight: .semibold, color: NSColor(white: 0.52, alpha: 1))
        logTitle.translatesAutoresizingMaskIntoConstraints = false

        let logBox = NSView()
        logBox.wantsLayer = true
        logBox.layer?.backgroundColor = NSColor(white: 0, alpha: 0.25).cgColor
        logBox.layer?.cornerRadius = 8
        logBox.translatesAutoresizingMaskIntoConstraints = false
        logBox.heightAnchor.constraint(equalToConstant: 92).isActive = true

        logView.isEditable = false
        logView.isSelectable = true
        logView.drawsBackground = false
        logView.textColor = NSColor(calibratedWhite: 0.66, alpha: 1)
        logView.font = NSFont.monospacedSystemFont(ofSize: 10.5, weight: .regular)
        logView.textContainerInset = NSSize(width: 7, height: 7)
        logView.isVerticallyResizable = true
        logView.isHorizontallyResizable = false
        logView.autoresizingMask = [.width]

        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.drawsBackground = false
        scroll.contentView.drawsBackground = false
        scroll.borderType = .noBorder
        scroll.documentView = logView
        scroll.translatesAutoresizingMaskIntoConstraints = false
        logBox.addSubview(scroll)
        NSLayoutConstraint.activate([
            scroll.leadingAnchor.constraint(equalTo: logBox.leadingAnchor, constant: 2),
            scroll.trailingAnchor.constraint(equalTo: logBox.trailingAnchor, constant: -2),
            scroll.topAnchor.constraint(equalTo: logBox.topAnchor, constant: 2),
            scroll.bottomAnchor.constraint(equalTo: logBox.bottomAnchor, constant: -2),
        ])

        pathLabel.font = NSFont.systemFont(ofSize: 9.5, weight: .regular)
        pathLabel.textColor = NSColor(white: 0.48, alpha: 1)
        pathLabel.lineBreakMode = .byTruncatingMiddle
        pathLabel.translatesAutoresizingMaskIntoConstraints = false

        let changeButton = NSButton(title: "Change…", target: self, action: #selector(changeRepo))
        changeButton.bezelStyle = .inline
        changeButton.isBordered = false
        changeButton.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        changeButton.contentTintColor = NSColor(white: 0.62, alpha: 1)
        changeButton.translatesAutoresizingMaskIntoConstraints = false

        let footer = NSStackView(views: [pathLabel, changeButton])
        footer.orientation = .horizontal
        footer.spacing = 10
        footer.alignment = .centerY

        let content = NSStackView(views: [header, card, buttonRow, openButton, logTitle, logBox, footer])
        content.orientation = .vertical
        content.alignment = .leading
        content.spacing = 12
        content.translatesAutoresizingMaskIntoConstraints = false
        content.setCustomSpacing(18, after: header)
        content.setCustomSpacing(16, after: card)
        content.setCustomSpacing(8, after: openButton)
        content.setCustomSpacing(5, after: logTitle)

        view.addSubview(content)
        NSLayoutConstraint.activate([
            content.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 26),
            content.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -26),
            content.topAnchor.constraint(equalTo: view.topAnchor, constant: 28),
            content.bottomAnchor.constraint(lessThanOrEqualTo: view.bottomAnchor, constant: -16),
        ])

        for item in [header, card, buttonRow, openButton, logBox, footer] {
            item.widthAnchor.constraint(equalTo: content.widthAnchor).isActive = true
        }
        footer.widthAnchor.constraint(equalTo: content.widthAnchor).isActive = true
    }

    private func render(_ state: ServerController.State) {
        statusLabel.stringValue = state.label
        detailLabel.stringValue = server.statusDetail

        let color: NSColor
        switch state {
        case .stopped: color = NSColor(white: 0.42, alpha: 1)
        case .starting, .stopping: color = NSColor(calibratedRed: 1.0, green: 0.76, blue: 0.25, alpha: 1)
        case .running: color = NSColor(calibratedRed: 0.30, green: 0.88, blue: 0.55, alpha: 1)
        case .error: color = NSColor(calibratedRed: 1.0, green: 0.39, blue: 0.38, alpha: 1)
        }
        statusDot.layer?.backgroundColor = color.cgColor

        if state.isBusy {
            spinner.startAnimation(nil)
        } else {
            spinner.stopAnimation(nil)
        }

        switch state {
        case .stopped, .error:
            startButton.isEnabled = true
            startButton.isHidden = false
        default:
            startButton.isEnabled = false
            startButton.isHidden = true
        }

        switch state {
        case .running, .starting:
            stopButton.isEnabled = true
            stopButton.isHidden = false
        case .stopping:
            stopButton.isEnabled = false
            stopButton.isHidden = false
        default:
            stopButton.isEnabled = false
            stopButton.isHidden = true
        }

        switch state {
        case .running:
            restartButton.isEnabled = true
            restartButton.isHidden = false
        default:
            restartButton.isEnabled = false
            restartButton.isHidden = true
        }

        openButton.isHidden = !state.isRunning
        openButton.isEnabled = state.isRunning
        applyButtonStyle(startButton, enabled: startButton.isEnabled, accent: startButton.isEnabled && !state.isRunning)
        applyButtonStyle(openButton, enabled: openButton.isEnabled, accent: state.isRunning)
        pathLabel.stringValue = abbreviatedPath(server.config.repoPath)
        pathLabel.toolTip = server.config.repoPath
    }

    private func abbreviatedPath(_ path: String) -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        if path.hasPrefix(home) {
            return "~" + path.dropFirst(home.count)
        }
        return path
    }

    private func appendLog(_ text: String) {
        logBuffer += text
        var pieces = logBuffer.components(separatedBy: "\n")
        logBuffer = pieces.removeLast()
        for piece in pieces where !piece.isEmpty {
            logLines.append(piece)
        }
        if logLines.count > 400 {
            logLines.removeFirst(logLines.count - 400)
        }
        logView.string = logLines.isEmpty ? "No server output yet." : logLines.joined(separator: "\n")
        logView.scrollToEndOfDocument(nil)
    }

    @objc private func startTapped() {
        server.start()
    }

    @objc private func stopTapped() {
        if let owner = server.portOwner(), !server.ownsRunningProcess {
            let alert = NSAlert()
            alert.messageText = "Stop the control center?"
            alert.informativeText = "The server on port \(server.config.port) was started outside LLM Controller (PID \(owner.pid)). Stopping it will end that process."
            alert.alertStyle = .warning
            alert.addButton(withTitle: "Stop Server")
            alert.addButton(withTitle: "Cancel")
            if alert.runModal() != .alertFirstButtonReturn {
                return
            }
        }
        server.stop()
    }

    @objc private func restartTapped() {
        server.restart()
    }

    @objc private func openTapped() {
        server.openDashboard()
    }

    @objc private func changeRepo() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Use Folder"
        panel.message = "Choose the macOS LLM Control Center folder (the one containing mlx_gui_server.py)."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard FileManager.default.fileExists(atPath: url.appendingPathComponent("mlx_gui_server.py").path) else {
            let alert = NSAlert()
            alert.messageText = "That folder doesn’t contain mlx_gui_server.py"
            alert.informativeText = "Pick the folder where the control center is installed."
            alert.runModal()
            return
        }
        server.updateRepo(url.path)
        pathLabel.stringValue = abbreviatedPath(url.path)
        pathLabel.toolTip = url.path
        detailLabel.stringValue = server.statusDetail
    }
}

// MARK: - App delegate

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var controller: ServerController!
    private var window: NSWindow!
    private var viewController: MainViewController!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        buildMenu()

        let config = AppConfig.load()
        controller = ServerController(config: config)
        viewController = MainViewController(server: controller)

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 460, height: 460),
            styleMask: [.titled, .closable, .miniaturizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "LLM Controller"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = true
        window.backgroundColor = NSColor(calibratedRed: 0.043, green: 0.055, blue: 0.094, alpha: 1)
        window.appearance = NSAppearance(named: .darkAqua)
        window.contentViewController = viewController
        window.isReleasedWhenClosed = false
        window.center()
        window.setFrameAutosaveName("MLXControllerWindow")
        window.makeKeyAndOrderFront(nil)
        self.window = window

        NSApp.activate(ignoringOtherApps: true)

        if let snapshotPath = ProcessInfo.processInfo.environment["MLX_CONTROLLER_SNAPSHOT"] {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.4) { [weak self] in
                guard let view = self?.window.contentView else { return }
                guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
                view.cacheDisplay(in: view.bounds, to: rep)
                if let data = rep.representation(using: .png, properties: [:]) {
                    try? data.write(to: URL(fileURLWithPath: snapshotPath))
                }
            }
        }

        controller.refresh { [weak self] in
            guard let self else { return }
            if let problem = self.controller.config.validationError() {
                self.controller.onStateChange?(.error(problem))
                self.promptForProject()
                return
            }
            if self.controller.config.autoStart, case .stopped = self.controller.state {
                self.controller.start()
            }
        }
    }

    private func promptForProject() {
        let alert = NSAlert()
        alert.messageText = "Where is LLM Control Center?"
        alert.informativeText = "Choose the folder that contains mlx_gui_server.py. MLX Controller will remember it."
        alert.addButton(withTitle: "Choose Folder…")
        alert.addButton(withTitle: "Later")
        guard alert.runModal() == .alertFirstButtonReturn else { return }

        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Use Folder"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        controller.updateRepo(url.path)
        controller.refresh { [weak self] in
            guard let self else { return }
            if self.controller.config.autoStart, case .stopped = self.controller.state {
                self.controller.start()
            }
        }
    }

    private func buildMenu() {
        let mainMenu = NSMenu()

        let appItem = NSMenuItem()
        mainMenu.addItem(appItem)
        let appMenu = NSMenu()
        let about = NSMenuItem(title: "About LLM Controller", action: #selector(showAbout), keyEquivalent: "")
        about.target = self
        appMenu.addItem(about)
        appMenu.addItem(.separator())
        let dashboard = NSMenuItem(title: "Open Dashboard", action: #selector(menuOpenDashboard), keyEquivalent: "d")
        dashboard.target = self
        appMenu.addItem(dashboard)
        let reveal = NSMenuItem(title: "Reveal Project Folder", action: #selector(menuRevealProject), keyEquivalent: "")
        reveal.target = self
        appMenu.addItem(reveal)
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Hide LLM Controller", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(withTitle: "Quit LLM Controller", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu

        let editItem = NSMenuItem()
        mainMenu.addItem(editItem)
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = editMenu

        NSApp.mainMenu = mainMenu
    }

    @objc private func showAbout() {
        let alert = NSAlert()
        alert.messageText = "LLM Controller"
        alert.informativeText = "A small desktop controller for the macOS LLM Control Center.\n\nStarts and stops the local server; the dashboard opens in your browser.\n\nVersion 1.0.5"
        alert.icon = NSApp.applicationIconImage
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }

    @objc private func menuOpenDashboard() {
        controller.openDashboard()
    }

    @objc private func menuRevealProject() {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: controller.config.repoPath)])
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        window.makeKeyAndOrderFront(nil)
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        controller.isQuitting = true
        if controller.config.stopOnQuit, controller.ownsRunningProcess {
            controller.stop {
                sender.reply(toApplicationShouldTerminate: true)
            }
            return .terminateLater
        }
        return .terminateNow
    }

    func applicationDockMenu(_ sender: NSApplication) -> NSMenu? {
        let menu = NSMenu()
        let open = NSMenuItem(title: "Open Dashboard", action: #selector(menuOpenDashboard), keyEquivalent: "")
        open.target = self
        menu.addItem(open)
        menu.addItem(.separator())
        let start = NSMenuItem(title: "Start Server", action: #selector(menuStart), keyEquivalent: "")
        start.target = self
        menu.addItem(start)
        let restart = NSMenuItem(title: "Restart Server", action: #selector(menuRestart), keyEquivalent: "")
        restart.target = self
        menu.addItem(restart)
        let stop = NSMenuItem(title: "Stop Server", action: #selector(menuStop), keyEquivalent: "")
        stop.target = self
        menu.addItem(stop)
        return menu
    }

    @objc private func menuStart() { controller.start() }
    @objc private func menuRestart() { controller.restart() }
    @objc private func menuStop() { controller.stop() }
}

// MARK: - Entry point

let application = NSApplication.shared
let appDelegate = AppDelegate()
application.delegate = appDelegate
application.run()
