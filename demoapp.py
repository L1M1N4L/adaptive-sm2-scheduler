"""
Braille Flashcard Demo Application
app_demo.py

A tkinter-based demo application for testing spaced repetition algorithms
with Braille pattern learning simulation.

Supports: SM-2, FSRS, and Hybrid algorithms

Usage:
    python app_demo.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys
import os
import json
import csv
import pickle

# Import from the actual scheduler files in your project
try:
    from src.schedulers.base import BaseScheduler, ScheduleDecision, RatingConverter
    from src.schedulers.sm2 import SM2Scheduler
    from src.schedulers.hybrid import HybridScheduler
    # Add these imports if you have FSRS implementation
    # from src.schedulers.fsrs import FSRSScheduler
except ImportError as e:
    print(f"Error importing schedulers: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


# Braille Patterns Database
BRAILLE_PATTERNS = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑',
    'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
    'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕',
    'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
    'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵'
}


class FlashcardApp:
    """Main flashcard application with multiple scheduler support."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Braille Flashcard Learning - Algorithm Comparison")
        self.root.geometry("900x650")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize all available schedulers
        self.schedulers = {
            'SM-2': SM2Scheduler(),
            # Uncomment these when you have the implementations
            # 'FSRS': FSRSScheduler(),
            # 'Hybrid': HybridScheduler()
        }
        
        # Check if FSRS and Hybrid are available (mock for now if not)
        try:
            from src.schedulers.fsrs import FSRSScheduler
            self.schedulers['FSRS'] = FSRSScheduler()
        except ImportError:
            print("FSRS not found, will use mock implementation")
            self.schedulers['FSRS'] = self._create_mock_fsrs()
        
        try:
            from src.schedulers.hybrid import HybridScheduler
            # Initialize hybrid with ML models enabled
            self.schedulers['Hybrid'] = HybridScheduler(
                use_hlr=True,
                use_dhp=True,
                use_rnn=False  # RNN requires PyTorch and trained model
            )
        except ImportError:
            print("Hybrid not found, will use mock implementation")
            self.schedulers['Hybrid'] = self._create_mock_hybrid()
        
        # Current algorithm selection
        self.current_algorithm = 'SM-2'
        self.scheduler = self.schedulers[self.current_algorithm]
        
        self.user_id = "demo_user"
        self.current_day = 0.0
        
        # Flashcard deck
        self.deck = list(BRAILLE_PATTERNS.keys())
        self.current_card = None
        self.show_answer = False
        
        # Due cards queue
        self.due_cards = []
        self.review_count = 0
        self.cards_learned = 0
        
        # Statistics per algorithm
        self.algorithm_stats = {
            'SM-2': {'total_reviews': 0, 'correct': 0, 'incorrect': 0, 'streak': 0},
            'FSRS': {'total_reviews': 0, 'correct': 0, 'incorrect': 0, 'streak': 0},
            'Hybrid': {'total_reviews': 0, 'correct': 0, 'incorrect': 0, 'streak': 0}
        }
        
        self.stats = self.algorithm_stats[self.current_algorithm]
        
        # Detailed review history for analysis
        self.review_history = []  # Store all reviews
        self.session_start = datetime.now()
        
        # Results directory
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.setup_ui()
        self.load_due_cards()
        self.next_card()
    
    def _create_mock_fsrs(self):
        """Create a mock FSRS scheduler if not available."""
        class MockFSRS(BaseScheduler):
            def schedule_review(self, user_id, item_id, rating, timestamp):
                state = self.get_state(user_id, item_id)
                # Simple FSRS-like logic
                if rating < 3:
                    interval = 1
                    repetitions = 0
                else:
                    repetitions = state.repetitions + 1
                    stability = max(1, state.interval * (1 + rating * 0.3))
                    interval = int(stability * 0.9)
                
                state.interval = interval
                state.repetitions = repetitions
                state.last_review = timestamp
                self.update_state(user_id, item_id, state)
                self.total_reviews += 1
                
                return ScheduleDecision(
                    interval=interval,
                    ease_factor=2.5,
                    repetitions=repetitions,
                    p_recall=0.9,
                    confidence=0.8
                )
            
            def predict_recall(self, user_id, item_id, delta_t):
                return 2 ** (-delta_t / 10)
            
            def calculate_half_life(self, user_id, item_id):
                state = self.get_state(user_id, item_id)
                return max(1.0, state.interval * 0.5)
        
        return MockFSRS()
    
    def _create_mock_hybrid(self):
        """Create a mock Hybrid scheduler if not available."""
        class MockHybrid(BaseScheduler):
            def __init__(self):
                super().__init__()
                self.beta = 0.5
            
            def schedule_review(self, user_id, item_id, rating, timestamp):
                state = self.get_state(user_id, item_id)
                
                # SM-2 component
                ef = state.ease_factor + (0.1 - (5 - rating) * (0.08 + (5 - rating) * 0.02))
                ef = max(1.3, ef)
                
                if rating < 3:
                    sm2_interval = 1
                    repetitions = 0
                else:
                    repetitions = state.repetitions + 1
                    if repetitions == 1:
                        sm2_interval = 1
                    elif repetitions == 2:
                        sm2_interval = 6
                    else:
                        sm2_interval = round(state.interval * ef)
                
                # ML component (simplified)
                ml_interval = max(1, int(sm2_interval * (1 + rating * 0.1)))
                
                # Blend with adaptive beta
                adaptive_beta = min(0.8, repetitions * 0.1)  # More ML as experience grows
                interval = int((1 - adaptive_beta) * sm2_interval + adaptive_beta * ml_interval)
                
                state.ease_factor = ef
                state.interval = interval
                state.repetitions = repetitions
                state.last_review = timestamp
                self.update_state(user_id, item_id, state)
                self.total_reviews += 1
                
                return ScheduleDecision(
                    interval=interval,
                    ease_factor=ef,
                    repetitions=repetitions,
                    p_recall=0.9,
                    confidence=adaptive_beta
                )
            
            def predict_recall(self, user_id, item_id, delta_t):
                return 2 ** (-delta_t / 12)
            
            def calculate_half_life(self, user_id, item_id):
                state = self.get_state(user_id, item_id)
                return max(1.0, state.interval * 0.6)
        
        return MockHybrid()
    
    def setup_ui(self):
        """Setup the user interface."""
        
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🔤 Braille Flashcard Learning",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Main container
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Statistics & Controls
        left_panel = tk.Frame(main_container, bg="white", relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Algorithm Selector
        algo_frame = tk.Frame(left_panel, bg="white")
        algo_frame.pack(pady=15, padx=20, fill=tk.X)
        
        tk.Label(
            algo_frame,
            text="🧠 Algorithm:",
            font=("Arial", 12, "bold"),
            bg="white"
        ).pack(anchor="w", pady=(0, 5))
        
        self.algorithm_var = tk.StringVar(value=self.current_algorithm)
        algorithm_menu = ttk.Combobox(
            algo_frame,
            textvariable=self.algorithm_var,
            values=list(self.schedulers.keys()),
            state="readonly",
            font=("Arial", 11)
        )
        algorithm_menu.pack(fill=tk.X)
        algorithm_menu.bind('<<ComboboxSelected>>', self.change_algorithm)
        
        # Separator
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=20, pady=10)
        
        # Statistics
        stats_title = tk.Label(
            left_panel,
            text="📊 Statistics",
            font=("Arial", 14, "bold"),
            bg="white"
        )
        stats_title.pack(pady=10, padx=20)
        
        self.stats_labels = {}
        stats_info = [
            ("Day:", "current_day"),
            ("Cards Due:", "due_cards"),
            ("Reviews Today:", "reviews_today"),
            ("Total Reviews:", "total_reviews"),
            ("Correct:", "correct"),
            ("Incorrect:", "incorrect"),
            ("Accuracy:", "accuracy"),
            ("Current Streak:", "streak")
        ]
        
        for label, key in stats_info:
            frame = tk.Frame(left_panel, bg="white")
            frame.pack(fill=tk.X, padx=20, pady=5)
            
            tk.Label(
                frame,
                text=label,
                font=("Arial", 10, "bold"),
                bg="white",
                anchor="w"
            ).pack(side=tk.LEFT)
            
            value_label = tk.Label(
                frame,
                text="0",
                font=("Arial", 10),
                bg="white",
                anchor="e"
            )
            value_label.pack(side=tk.RIGHT)
            self.stats_labels[key] = value_label
        
        # Separator
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=20, pady=10)
        
        # Comparison button
        tk.Button(
            left_panel,
            text="📈 Compare Algorithms",
            font=("Arial", 11),
            command=self.show_comparison,
            bg="#9b59b6",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=10
        ).pack(pady=10, padx=20, fill=tk.X)
        
        # Advance day button
        tk.Button(
            left_panel,
            text="⏭️ Next Day",
            font=("Arial", 11),
            command=self.advance_day,
            bg="#3498db",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=10
        ).pack(pady=10, padx=20, fill=tk.X)
        
        # Reset button
        tk.Button(
            left_panel,
            text="🔄 Reset All",
            font=("Arial", 10),
            command=self.reset_all,
            bg="#e74c3c",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=8
        ).pack(pady=10, padx=20, fill=tk.X)
        
        # Separator
        ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=20, pady=10)
        
        # Export buttons
        tk.Label(
            left_panel,
            text="💾 Export Results",
            font=("Arial", 12, "bold"),
            bg="white"
        ).pack(pady=(10, 5), padx=20)
        
        tk.Button(
            left_panel,
            text="📊 Export CSV",
            font=("Arial", 10),
            command=self.export_csv,
            bg="#16a085",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=8
        ).pack(pady=5, padx=20, fill=tk.X)
        
        tk.Button(
            left_panel,
            text="📄 Export JSON",
            font=("Arial", 10),
            command=self.export_json,
            bg="#16a085",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=8
        ).pack(pady=5, padx=20, fill=tk.X)
        
        tk.Button(
            left_panel,
            text="💿 Save Session",
            font=("Arial", 10),
            command=self.save_session,
            bg="#8e44ad",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=8
        ).pack(pady=5, padx=20, fill=tk.X)
        
        tk.Button(
            left_panel,
            text="📂 Load Session",
            font=("Arial", 10),
            command=self.load_session,
            bg="#8e44ad",
            fg="white",
            relief=tk.FLAT,
            padx=20,
            pady=8
        ).pack(pady=5, padx=20, fill=tk.X)
        
        # Right panel - Flashcard
        right_panel = tk.Frame(main_container, bg="#f0f0f0")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Algorithm indicator
        self.algo_indicator = tk.Label(
            right_panel,
            text=f"Using: {self.current_algorithm}",
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            relief=tk.RAISED,
            padx=15,
            pady=8
        )
        self.algo_indicator.pack(pady=(0, 10))
        
        # Card display
        card_frame = tk.Frame(right_panel, bg="white", relief=tk.RAISED, borderwidth=3)
        card_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Question/Answer display
        self.card_label = tk.Label(
            card_frame,
            text="",
            font=("Arial", 80),
            bg="white",
            fg="#2c3e50"
        )
        self.card_label.pack(expand=True)
        
        self.letter_label = tk.Label(
            card_frame,
            text="",
            font=("Arial", 24),
            bg="white",
            fg="#7f8c8d"
        )
        self.letter_label.pack(pady=10)
        
        # Info label for scheduling info
        self.info_label = tk.Label(
            card_frame,
            text="",
            font=("Arial", 10),
            bg="white",
            fg="#95a5a6"
        )
        self.info_label.pack(pady=5)
        
        # Show answer button
        self.show_answer_btn = tk.Button(
            right_panel,
            text="👁️ Show Answer",
            font=("Arial", 14, "bold"),
            command=self.toggle_answer,
            bg="#2ecc71",
            fg="white",
            relief=tk.FLAT,
            padx=30,
            pady=15
        )
        self.show_answer_btn.pack(pady=(0, 20))
        
        # Rating buttons frame
        self.rating_frame = tk.Frame(right_panel, bg="#f0f0f0")
        self.rating_frame.pack(fill=tk.X)
        
        rating_buttons = [
            ("❌ Again\n(Hard)", 0, "#e74c3c"),
            ("😕 Hard\n(Medium)", 3, "#e67e22"),
            ("😊 Good\n(Easy)", 4, "#2ecc71"),
            ("🎯 Perfect\n(Very Easy)", 5, "#3498db")
        ]
        
        for text, rating, color in rating_buttons:
            btn = tk.Button(
                self.rating_frame,
                text=text,
                font=("Arial", 11),
                command=lambda r=rating: self.rate_card(r),
                bg=color,
                fg="white",
                relief=tk.FLAT,
                padx=15,
                pady=15,
                state=tk.DISABLED
            )
            btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        self.update_stats()
    
    def change_algorithm(self, event=None):
        """Change the active scheduling algorithm."""
        new_algorithm = self.algorithm_var.get()
        
        if new_algorithm == self.current_algorithm:
            return
        
        confirm = messagebox.askyesno(
            "Change Algorithm",
            f"Switch from {self.current_algorithm} to {new_algorithm}?\n\n"
            f"This will start a new learning session with {new_algorithm}."
        )
        
        if confirm:
            self.current_algorithm = new_algorithm
            self.scheduler = self.schedulers[new_algorithm]
            self.stats = self.algorithm_stats[new_algorithm]
            self.algo_indicator.config(text=f"Using: {self.current_algorithm}")
            
            self.load_due_cards()
            self.update_stats()
            self.next_card()
    
    def load_due_cards(self):
        """Load cards that are due for review."""
        self.due_cards = []
        
        # Check all cards
        for letter in self.deck:
            item_id = f"braille_{letter}"
            state = self.scheduler.get_state(self.user_id, item_id)
            
            # New cards or cards due today
            if state.repetitions == 0:
                self.due_cards.append(letter)
            elif state.last_review + state.interval <= self.current_day:
                self.due_cards.append(letter)
        
        # Limit to 10 new cards per day
        new_cards = [c for c in self.due_cards if self.scheduler.get_state(
            self.user_id, f"braille_{c}").repetitions == 0]
        review_cards = [c for c in self.due_cards if c not in new_cards]
        
        self.due_cards = review_cards + new_cards[:10]
        random.shuffle(self.due_cards)
    
    def next_card(self):
        """Show the next card."""
        if not self.due_cards:
            messagebox.showinfo(
                "All Done! 🎉",
                f"You've reviewed all cards for today!\n\n"
                f"Algorithm: {self.current_algorithm}\n"
                f"Reviews: {self.review_count}\n"
                f"Come back tomorrow for more practice!"
            )
            self.card_label.config(text="✅")
            self.letter_label.config(text="All cards reviewed!")
            self.info_label.config(text="")
            self.show_answer_btn.config(state=tk.DISABLED)
            return
        
        self.current_card = self.due_cards[0]
        self.show_answer = False
        
        # Show Braille pattern
        self.card_label.config(text=BRAILLE_PATTERNS[self.current_card])
        self.letter_label.config(text="What letter is this?", fg="#7f8c8d", font=("Arial", 24))
        
        # Show card info
        item_id = f"braille_{self.current_card}"
        state = self.scheduler.get_state(self.user_id, item_id)
        if state.repetitions > 0:
            self.info_label.config(
                text=f"Reviews: {state.repetitions} | Ease: {state.ease_factor:.2f} | Last: {int(self.current_day - state.last_review)}d ago"
            )
        else:
            self.info_label.config(text="New card")
        
        self.show_answer_btn.config(state=tk.NORMAL, bg="#2ecc71", text="👁️ Show Answer")
        
        # Disable rating buttons
        for widget in self.rating_frame.winfo_children():
            widget.config(state=tk.DISABLED)
    
    def toggle_answer(self):
        """Toggle between question and answer."""
        if not self.show_answer:
            self.show_answer = True
            self.letter_label.config(
                text=f"Letter: {self.current_card.upper()}",
                fg="#2c3e50",
                font=("Arial", 32, "bold")
            )
            self.show_answer_btn.config(text="🔄 Hide Answer", bg="#e67e22")
            
            # Enable rating buttons
            for widget in self.rating_frame.winfo_children():
                widget.config(state=tk.NORMAL)
        else:
            self.show_answer = False
            self.letter_label.config(
                text="What letter is this?",
                fg="#7f8c8d",
                font=("Arial", 24)
            )
            self.show_answer_btn.config(text="👁️ Show Answer", bg="#2ecc71")
            
            # Disable rating buttons
            for widget in self.rating_frame.winfo_children():
                widget.config(state=tk.DISABLED)
    
    def rate_card(self, rating: int):
        """Rate the current card and schedule next review."""
        if not self.current_card:
            return
        
        item_id = f"braille_{self.current_card}"
        
        # Schedule review using current algorithm
        decision = self.scheduler.schedule_review(
            self.user_id,
            item_id,
            rating,
            self.current_day
        )
        
        # Update statistics
        self.stats['total_reviews'] += 1
        self.review_count += 1
        
        if rating >= 3:
            self.stats['correct'] += 1
            self.stats['streak'] += 1
        else:
            self.stats['incorrect'] += 1
            self.stats['streak'] = 0
        
        # Record detailed review history
        review_record = {
            'timestamp': datetime.now().isoformat(),
            'day': self.current_day,
            'algorithm': self.current_algorithm,
            'user_id': self.user_id,
            'item_id': item_id,
            'letter': self.current_card,
            'braille_pattern': BRAILLE_PATTERNS[self.current_card],
            'rating': rating,
            'rating_label': self._get_rating_label(rating),
            'interval': decision.interval,
            'ease_factor': decision.ease_factor,
            'repetitions': decision.repetitions,
            'p_recall': decision.p_recall,
            'confidence': decision.confidence if hasattr(decision, 'confidence') else 0.0,
            'was_correct': rating >= 3,
            'streak': self.stats['streak']
        }
        self.review_history.append(review_record)
        
        # Auto-save every 10 reviews
        if len(self.review_history) % 10 == 0:
            self._auto_save()
        
        # Show scheduling result
        confidence_text = f"\nConfidence: {decision.confidence:.1%}" if hasattr(decision, 'confidence') and decision.confidence > 0 else ""
        
        messagebox.showinfo(
            f"Scheduled! ✅ [{self.current_algorithm}]",
            f"Letter: {self.current_card.upper()}\n\n"
            f"Next review: {decision.interval} days\n"
            f"Ease factor: {decision.ease_factor:.2f}\n"
            f"Repetitions: {decision.repetitions}\n"
            f"Predicted recall: {decision.p_recall:.1%}"
            f"{confidence_text}"
        )
        
        # Remove from due cards and show next
        self.due_cards.pop(0)
        self.update_stats()
        self.next_card()
    
    def _get_rating_label(self, rating: int) -> str:
        """Get label for rating value."""
        labels = {0: 'Again', 3: 'Hard', 4: 'Good', 5: 'Perfect'}
        return labels.get(rating, 'Unknown')
    
    def _auto_save(self):
        """Auto-save review history."""
        try:
            filename = os.path.join(self.results_dir, 'autosave_reviews.json')
            with open(filename, 'w') as f:
                json.dump({
                    'review_history': self.review_history,
                    'algorithm_stats': self.algorithm_stats,
                    'current_day': self.current_day
                }, f, indent=2)
        except Exception as e:
            print(f"Auto-save failed: {e}")
    
    def export_csv(self):
        """Export review history to CSV."""
        if not self.review_history:
            messagebox.showwarning("No Data", "No review data to export!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.results_dir,
            initialfile=f"braille_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                if self.review_history:
                    writer = csv.DictWriter(f, fieldnames=self.review_history[0].keys())
                    writer.writeheader()
                    writer.writerows(self.review_history)
            
            messagebox.showinfo("Export Success", f"Exported {len(self.review_history)} reviews to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV:\n{str(e)}")
    
    def export_json(self):
        """Export complete session data to JSON."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self.results_dir,
            initialfile=f"braille_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if not filename:
            return
        
        try:
            # Collect all scheduler states
            scheduler_states = {}
            for algo_name, scheduler in self.schedulers.items():
                scheduler_states[algo_name] = {
                    'total_reviews': scheduler.total_reviews,
                    'total_items': len(scheduler.states),
                    'states': {}
                }
                
                # Export individual item states
                for (user_id, item_id), state in scheduler.states.items():
                    scheduler_states[algo_name]['states'][item_id] = {
                        'ease_factor': state.ease_factor,
                        'repetitions': state.repetitions,
                        'interval': state.interval,
                        'last_review': state.last_review
                    }
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'session_start': self.session_start.isoformat(),
                'current_day': self.current_day,
                'current_algorithm': self.current_algorithm,
                'algorithm_stats': self.algorithm_stats,
                'review_history': self.review_history,
                'scheduler_states': scheduler_states,
                'metadata': {
                    'total_reviews': len(self.review_history),
                    'algorithms_used': list(self.schedulers.keys()),
                    'cards_in_deck': len(self.deck)
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Export Success", f"Session data exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export JSON:\n{str(e)}")
    
    def save_session(self):
        """Save complete session state (including scheduler objects)."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=self.results_dir,
            initialfile=f"braille_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        
        if not filename:
            return
        
        try:
            session_data = {
                'schedulers': self.schedulers,
                'algorithm_stats': self.algorithm_stats,
                'review_history': self.review_history,
                'current_day': self.current_day,
                'current_algorithm': self.current_algorithm,
                'session_start': self.session_start
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(session_data, f)
            
            messagebox.showinfo("Save Success", f"Session saved to:\n{filename}\n\nYou can load this later to continue.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save session:\n{str(e)}")
    
    def load_session(self):
        """Load a saved session."""
        filename = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=self.results_dir,
            title="Load Session"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'rb') as f:
                session_data = pickle.load(f)
            
            # Restore session
            self.schedulers = session_data['schedulers']
            self.algorithm_stats = session_data['algorithm_stats']
            self.review_history = session_data['review_history']
            self.current_day = session_data['current_day']
            self.current_algorithm = session_data['current_algorithm']
            self.session_start = session_data.get('session_start', datetime.now())
            
            # Update UI
            self.scheduler = self.schedulers[self.current_algorithm]
            self.stats = self.algorithm_stats[self.current_algorithm]
            self.algorithm_var.set(self.current_algorithm)
            self.algo_indicator.config(text=f"Using: {self.current_algorithm}")
            
            self.load_due_cards()
            self.update_stats()
            self.next_card()
            
            messagebox.showinfo("Load Success", f"Session loaded!\n\nReviews: {len(self.review_history)}\nDay: {int(self.current_day)}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load session:\n{str(e)}")
    
    def advance_day(self):
        """Advance to the next day."""
        self.current_day += 1
        self.review_count = 0
        self.load_due_cards()
        self.update_stats()
        self.next_card()
        
        messagebox.showinfo(
            "New Day! 🌅",
            f"Day {int(self.current_day)}\n\n"
            f"Algorithm: {self.current_algorithm}\n"
            f"Cards due today: {len(self.due_cards)}"
        )
    
    def reset_all(self):
        """Reset all data."""
        confirm = messagebox.askyesno(
            "Reset All Data",
            "This will reset all progress for ALL algorithms.\n\n"
            "Are you sure?"
        )
        
        if confirm:
            for scheduler in self.schedulers.values():
                scheduler.reset()
            
            for algo in self.algorithm_stats:
                self.algorithm_stats[algo] = {
                    'total_reviews': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'streak': 0
                }
            
            self.current_day = 0.0
            self.review_count = 0
            self.stats = self.algorithm_stats[self.current_algorithm]
            
            self.load_due_cards()
            self.update_stats()
            self.next_card()
            
            messagebox.showinfo("Reset Complete", "All data has been reset!")
    
    def show_comparison(self):
        """Show comparison window for all algorithms."""
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Algorithm Comparison")
        comparison_window.geometry("600x400")
        comparison_window.configure(bg="white")
        
        # Title
        tk.Label(
            comparison_window,
            text="📊 Algorithm Performance Comparison",
            font=("Arial", 16, "bold"),
            bg="white"
        ).pack(pady=20)
        
        # Create comparison table
        frame = tk.Frame(comparison_window, bg="white")
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        headers = ["Algorithm", "Total Reviews", "Correct", "Incorrect", "Accuracy", "Streak"]
        for col, header in enumerate(headers):
            tk.Label(
                frame,
                text=header,
                font=("Arial", 11, "bold"),
                bg="#ecf0f1",
                relief=tk.RIDGE,
                padx=10,
                pady=5
            ).grid(row=0, column=col, sticky="nsew", padx=1, pady=1)
        
        # Add data rows
        for row, (algo, stats) in enumerate(self.algorithm_stats.items(), start=1):
            total = stats['total_reviews']
            correct = stats['correct']
            accuracy = (correct / total * 100) if total > 0 else 0
            
            values = [
                algo,
                str(total),
                str(correct),
                str(stats['incorrect']),
                f"{accuracy:.1f}%",
                str(stats['streak'])
            ]
            
            bg_color = "#d5f4e6" if algo == self.current_algorithm else "white"
            
            for col, value in enumerate(values):
                tk.Label(
                    frame,
                    text=value,
                    font=("Arial", 10),
                    bg=bg_color,
                    relief=tk.RIDGE,
                    padx=10,
                    pady=5
                ).grid(row=row, column=col, sticky="nsew", padx=1, pady=1)
        
        # Configure grid weights
        for col in range(len(headers)):
            frame.grid_columnconfigure(col, weight=1)
        
        # Close button
        tk.Button(
            comparison_window,
            text="Close",
            command=comparison_window.destroy,
            bg="#3498db",
            fg="white",
            relief=tk.FLAT,
            padx=30,
            pady=10
        ).pack(pady=20)
    
    def update_stats(self):
        """Update statistics display."""
        total = self.stats['total_reviews']
        correct = self.stats['correct']
        accuracy = (correct / total * 100) if total > 0 else 0
        
        self.stats_labels['current_day'].config(text=f"{int(self.current_day)}")
        self.stats_labels['due_cards'].config(text=f"{len(self.due_cards)}")
        self.stats_labels['reviews_today'].config(text=f"{self.review_count}")
        self.stats_labels['total_reviews'].config(text=f"{total}")
        self.stats_labels['correct'].config(text=f"{correct}")
        self.stats_labels['incorrect'].config(text=f"{self.stats['incorrect']}")
        self.stats_labels['accuracy'].config(text=f"{accuracy:.1f}%")
        self.stats_labels['streak'].config(text=f"{self.stats['streak']}")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = FlashcardApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()