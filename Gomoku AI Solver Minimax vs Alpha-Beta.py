import numpy as np  # Library for array handling
import random       # Library for random number generation
from tkinter import *          # GUI library
from tkinter import messagebox  # For popup message dialogs (info, alerts, etc.)

# Game constants
BOARD_DIM = 15  # Board size 15x15
EMPTY, BLACK, WHITE = 0, 1, -1  # Represent empty cell, black piece (human), white piece (AI)
AI_DEPTH = 2  # Depth for AI search algorithms

def clear_widgets(window):
    # Clear all widgets inside the given window, to refresh UI easily
    for widget in window.winfo_children():
        widget.destroy()

def build_main_menu(root):
    # Build the main menu UI
    clear_widgets(root)  # Clear existing widgets
    root.configure(bg='#228B22')  # Set window background to dark green

    # Create a canvas for possible decorations (not used much here)
    canvas = Canvas(root, width=600, height=600, bg='#228B22', highlightthickness=0)
    canvas.pack()  # Fill the window

    # Display the title "Gomoku" centered at the top with large white font
    Label(root, text="Gomoku", font=("Verdana", 32, "bold"), bg='#228B22', fg='white').place(relx=0.5, rely=0.15, anchor="center")

    # Button styling dictionary for reuse
    btn_style = {
        "font": ("Verdana", 16, "bold"),
        "width": 20,
        "height": 2,
        "cursor": "hand2",  # Hand cursor on hover
        "bd": 0,
        "relief": "ridge"
    }

    # Button to start Human vs AI game mode
    Button(root, text="Play vs AI", bg="#f0f0f0", fg="#333", activebackground="#444", activeforeground="#fff",
           command=lambda: GomokuUI(root, ai_vs_ai=False), **btn_style).place(relx=0.5, rely=0.45, anchor="center")

    # Button to start AI vs AI mode (watch two AIs play)
    Button(root, text="Watch AI vs AI", bg="#333", fg="#fff", activebackground="#fff", activeforeground="#000",
           command=lambda: GomokuUI(root, ai_vs_ai=True), **btn_style).place(relx=0.5, rely=0.65, anchor="center")

def init_board():
    # Initialize the game board as a 15x15 numpy array filled with EMPTY cells
    return np.zeros((BOARD_DIM, BOARD_DIM), dtype=int)

def get_valid_moves(board):
    # Generate all valid moves on the board (cells adjacent to any existing pieces)
    possible_moves = set()
    for r in range(BOARD_DIM):
        for c in range(BOARD_DIM):
            if board[r, c] != EMPTY:
                # Check neighboring cells (8 directions)
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < BOARD_DIM and 0 <= nc < BOARD_DIM and board[nr, nc] == EMPTY:
                            possible_moves.add((nr, nc))
    # If no pieces on board, return center position
    return list(possible_moves) if possible_moves else [(BOARD_DIM//2, BOARD_DIM//2)]

def has_winner(board, player):
    # Check if the given player has won by having 5 in a row
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # vertical, horizontal, diagonal right-down, diagonal right-up
    for dr, dc in directions:
        for r in range(BOARD_DIM):
            for c in range(BOARD_DIM):
                try:
                    # Check 5 consecutive positions in this direction
                    if all(board[r + dr*i, c + dc*i] == player for i in range(5)):
                        return True
                except IndexError:
                    continue  # Ignore out-of-bounds errors
    return False

def count_lines(board, player, length):
    # Count number of lines of specific length for the player with at least one open end
    count = 0
    for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        for r in range(BOARD_DIM):
            for c in range(BOARD_DIM):
                try:
                    line = [board[r + dr*i, c + dc*i] for i in range(length)]
                    # Ensure all cells belong to player and no opponent pieces in line
                    if line.count(player) == length and line.count(-player) == 0:
                        # Check if either end is empty (open)
                        pre = (r-dr, c-dc)
                        post = (r+dr*length, c+dc*length)
                        pre_ok = 0 <= pre[0] < BOARD_DIM and 0 <= pre[1] < BOARD_DIM and board[pre] == EMPTY
                        post_ok = 0 <= post[0] < BOARD_DIM and 0 <= post[1] < BOARD_DIM and board[post] == EMPTY
                        if pre_ok or post_ok:
                            count += 1
                except IndexError:
                    continue
    return count

def score_board(board, ai_player):
    # Heuristic evaluation function to score the board state from AI player's perspective
    if has_winner(board, ai_player):
        return 1_000_000  # Max score if AI won
    if has_winner(board, -ai_player):
        return -1_000_000  # Min score if opponent won

    # Weights for lines of different lengths
    weights = {"four": 10000, "three": 1000, "two": 100}
    score = (
        weights["four"] * count_lines(board, ai_player, 4) -
        weights["four"] * 1.2 * count_lines(board, -ai_player, 4) +
        weights["three"] * count_lines(board, ai_player, 3) -
        weights["three"] * 1.5 * count_lines(board, -ai_player, 3) +
        weights["two"] * count_lines(board, ai_player, 2) -
        weights["two"] * 1.2 * count_lines(board, -ai_player, 2)
    )

    # Bonus for pieces closer to center (better control)
    center = BOARD_DIM // 2
    for r in range(BOARD_DIM):
        for c in range(BOARD_DIM):
            weight = (7 - abs(center - r)) * (7 - abs(center - c))
            if board[r, c] == ai_player:
                score += 2 * weight
            elif board[r, c] == -ai_player:
                score -= 2 * weight
    return score

def alpha_beta(board, depth, alpha, beta, maximize, ai_player, current):
    # Alpha-Beta pruning search algorithm to decide best move for AI
    if depth == 0 or has_winner(board, ai_player) or has_winner(board, -ai_player) or not get_valid_moves(board):
        return score_board(board, ai_player), None

    best_moves = []
    best_score = -float('inf') if maximize else float('inf')
    moves = get_valid_moves(board)
    move_scores = []

    # Evaluate moves heuristically to sort and reduce branching
    for move in moves:
        board[move] = current
        score = score_board(board, ai_player)
        board[move] = EMPTY
        move_scores.append((score, move))

    # Sort moves based on evaluation (descending if maximizing)
    move_scores.sort(reverse=maximize)
    moves = [m for _, m in move_scores[:8]]  # Limit to top 8 moves
    random.shuffle(moves)  # Shuffle to introduce randomness

    # Search through moves recursively
    for move in moves:
        board[move] = current
        eval_score, _ = alpha_beta(board, depth-1, alpha, beta, not maximize, ai_player, -current)
        board[move] = EMPTY

        if maximize:
            if eval_score > best_score:
                best_score = eval_score
                best_moves = [move]
            elif eval_score == best_score:
                best_moves.append(move)
            alpha = max(alpha, best_score)
        else:
            if eval_score < best_score:
                best_score = eval_score
                best_moves = [move]
            elif eval_score == best_score:
                best_moves.append(move)
            beta = min(beta, best_score)

        if beta <= alpha:
            break  # Prune remaining branches

    return best_score, random.choice(best_moves) if best_moves else None

def minimax(board, depth, maximize, ai_player, current):
    # Simple minimax search without alpha-beta pruning
    if depth == 0 or has_winner(board, ai_player) or has_winner(board, -ai_player) or not get_valid_moves(board):
        return score_board(board, ai_player), None

    best_score = -float('inf') if maximize else float('inf')
    best_moves = []
    moves = get_valid_moves(board)

    for move in moves:
        board[move] = current
        eval_score, _ = minimax(board, depth-1, not maximize, ai_player, -current)
        board[move] = EMPTY

        if maximize:
            if eval_score > best_score:
                best_score = eval_score
                best_moves = [move]
            elif eval_score == best_score:
                best_moves.append(move)
        else:
            if eval_score < best_score:
                best_score = eval_score
                best_moves = [move]
            elif eval_score == best_score:
                best_moves.append(move)

    return best_score, random.choice(best_moves) if best_moves else None

class GomokuUI:
    def __init__(self, root, ai_vs_ai=False):
        # Initialize the game UI, root is main window, ai_vs_ai indicates mode
        self.root = root
        clear_widgets(root)  # Clear any existing UI widgets
        self.board = init_board()  # Create empty board
        self.current_player = BLACK  # Human (Black) always starts first
        self.ai_mode = ai_vs_ai

        # Define AI types for black and white pieces
        self.ai_black = "Minimax"
        self.ai_white = "Alpha-Beta" if ai_vs_ai else "AI"  # If AI vs AI, white is alpha-beta, else just "AI"

        # Set window title to indicate mode
        title_mode = "AI vs AI" if ai_vs_ai else "Human (X) vs AI (O)"
        self.root.title(f"Gomoku - {title_mode}")

        # Create drawing canvas for the board
        self.canvas = Canvas(root, width=600, height=600, bg='#228B22', highlightthickness=0)
        self.canvas.pack()
        self.cell = 600 // BOARD_DIM  # Size of each cell in pixels
        self.canvas.bind("<Button-1>", self.player_move)  # Bind left-click to player move function

        # Status label showing whose turn it is
        status_text = f"X's turn | Human" if self.current_player == BLACK else f"O's turn | {self.ai_white}"
        self.status = Label(root, text=status_text, font=("Verdana", 14), bg='#228B22', fg='white')
        self.status.pack(fill=X)

        self.draw_board()  # Draw initial empty board

        # If AI vs AI, start the AI loop immediately
        if ai_vs_ai:
            self.current_player = BLACK  # Black AI starts first
            self.status.config(text=f"X ({self.ai_black}) thinking...")
            self.root.after(100, self.run_ai_turn)
        elif self.current_player == WHITE:
            # If AI starts first (not the case here), run AI turn
            self.status.config(text=f"O ({self.ai_white}) thinking...")
            self.root.after(100, self.run_ai_turn)

    def draw_board(self):
        # Clear and redraw the board grid and pieces
        self.canvas.delete("all")
        line_color = "#000000"  # Black grid lines

        # Draw grid lines
        for i in range(BOARD_DIM):
            # Horizontal lines
            self.canvas.create_line(self.cell//2, self.cell//2 + i*self.cell,
                                    self.cell//2 + (BOARD_DIM-1)*self.cell, self.cell//2 + i*self.cell,
                                    fill=line_color, width=2)
            # Vertical lines
            self.canvas.create_line(self.cell//2 + i*self.cell, self.cell//2,
                                    self.cell//2 + i*self.cell, self.cell//2 + (BOARD_DIM-1)*self.cell,
                                    fill=line_color, width=2)

        # Draw pieces (X and O)
        for r in range(BOARD_DIM):
            for c in range(BOARD_DIM):
                piece = self.board[r, c]
                if piece != EMPTY:
                    x_center = c * self.cell + self.cell // 2
                    y_center = r * self.cell + self.cell // 2
                    if piece == BLACK:
                        # Draw 'X' in white color inside cell
                        offset = self.cell // 3
                        self.canvas.create_line(x_center - offset, y_center - offset,
                                                x_center + offset, y_center + offset,
                                                fill="white", width=3)
                        self.canvas.create_line(x_center - offset, y_center + offset,
                                                x_center + offset, y_center - offset,
                                                fill="white", width=3)
                    else:
                        # Draw 'O' as a white circle inside cell
                        radius = self.cell // 3
                        self.canvas.create_oval(x_center - radius, y_center - radius,
                                                x_center + radius, y_center + radius,
                                                outline="white", width=3)

    def player_move(self, event):
        # Handle player's click to place a piece
        if self.ai_mode or self.current_player != BLACK:
            return  # Ignore clicks if AI mode or not human's turn (black)
        c, r = event.x // self.cell, event.y // self.cell  # Get clicked cell
        if 0 <= r < BOARD_DIM and 0 <= c < BOARD_DIM and self.board[r, c] == EMPTY:
            self.board[r, c] = BLACK  # Place black piece
            self.draw_board()  # Redraw board with new piece
            if has_winner(self.board, BLACK):
                self.declare_winner("X (Human)")  # Declare human win
            else:
                self.current_player = WHITE  # Switch to AI turn
                self.status.config(text=f"O ({self.ai_white}) thinking...")
                self.root.after(100, self.run_ai_turn)  # Run AI turn shortly

    def run_ai_turn(self):
        # Handle AI's move computation and placing piece
        if has_winner(self.board, -self.current_player):
            return  # Game over, do nothing
        if not get_valid_moves(self.board):
            # No moves left, it's a draw
            self.status.config(text="Draw!")
            messagebox.showinfo("Game Over", "It's a draw!")
            build_main_menu(self.root)
            return

        # Determine which AI to use based on current player
        ai_type = self.ai_black if self.current_player == BLACK else self.ai_white
        self.status.config(text=f"{'X' if self.current_player == BLACK else 'O'} ({ai_type}) thinking...")

        # Choose AI algorithm and get best move
        if ai_type == "Minimax":
            _, move = minimax(self.board, AI_DEPTH, True, self.current_player, self.current_player)
        else:
            # Default to Alpha-Beta
            _, move = alpha_beta(self.board, AI_DEPTH, -float('inf'), float('inf'), True, self.current_player, self.current_player)

        if move is not None:
            self.board[move] = self.current_player
            self.draw_board()
            if has_winner(self.board, self.current_player):
                winner = "X (AI Black)" if self.current_player == BLACK else "O (AI White)"
                self.declare_winner(winner)
                return
            self.current_player = -self.current_player  # Switch turns

            # Update status for next player
            if self.ai_mode or self.current_player == WHITE:
                # Continue AI vs AI or AI turn
                self.status.config(text=f"{'X' if self.current_player == BLACK else 'O'} ({self.ai_black if self.current_player == BLACK else self.ai_white}) thinking...")
                self.root.after(100, self.run_ai_turn)
            else:
                # Human's turn
                self.status.config(text=f"X's turn | Human")

    def declare_winner(self, winner):
        # Show message box announcing winner and return to main menu
        messagebox.showinfo("Game Over", f"{winner} wins!")
        build_main_menu(self.root)

if __name__ == "__main__":
    # Run the app
    root = Tk()
    root.geometry("600x680")
    root.resizable(False, False)
    build_main_menu(root)
    root.mainloop()
