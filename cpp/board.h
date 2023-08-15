class Board
{
private:
    enum CHESS_BASE_INFO
    {
        BLACK_CHESS     = 0,
        WHITE_CHESS     = 1,
        WIDTH           = 8,
        HEIGHT          = 8,
    };

    const static int width = WIDTH;
    const static int height = HEIGHT;
    int board[width][height] = {0};

public:
    Board()
    {
        //Init();
    }

    void Init();
    void CheckPosValid(int type, int x, int y);
    void SetChess(int type, int x, int y);

};