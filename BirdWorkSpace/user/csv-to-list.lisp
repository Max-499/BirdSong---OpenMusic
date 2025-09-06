(defun split-string (string delimiter)
  "Splits STRING into substrings at every occurrence of DELIMITER (a character)."
  (let ((start 0)
        (result '()))
    (loop for pos = (position delimiter string :start start)
          do (if pos
                 (progn
                   (push (subseq string start pos) result)
                   (setf start (1+ pos)))
                 (progn
                   (push (subseq string start) result)
                   (return))))
    (nreverse result)))

(defun read-csv (filepath)
  "Reads a 2-column CSV file with header. Returns a list of (x y) pairs."
  (with-open-file (in filepath)
    (let ((lines '()))
      ;; Skip header line
      (read-line in nil)
      ;; Read remaining lines
      (loop for line = (read-line in nil)
            while line
            do (let* ((split (split-string line #\,))
                      (x (read-from-string (first split)))
                      (y (read-from-string (second split))))
                 (push (list x y) lines)))
      (reverse lines))))

(defun csv-to-xy-lists (filepath)
  "Reads CSV and returns two lists: (x-list y-list)."
  (let* ((pairs (read-csv filepath))
         (xs (mapcar #'first pairs))
         (ys (mapcar #'second pairs)))
    (values xs ys)))

;; Example usage:
;; (csv-to-xy-lists "/Users/you/Desktop/birds_normalized_fullrange.csv")
;; => returns two lists, one for x and one for y
